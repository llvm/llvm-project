#include "inter/Analysis/MemoryFrontierAnalysis.h"
#include "inter/Dialect/Inter/IR/XW.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "inter/Transforms/Passes.h"

namespace inter {
#define GEN_PASS_DEF_INFERMEMORYTOKENS
#include "inter/Transforms/Passes.h.inc"
} // namespace inter.

using namespace mlir;

namespace {

static bool isXWMemoryOperation(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "xw.load" || name == "xw.store" ||
         name == "xw.atomic_rmw" || name == "xw.barrier";
}

static SmallVector<MemoryEffects::EffectInstance> getEffects(Operation *op) {
  SmallVector<MemoryEffects::EffectInstance> effects;
  if (auto memory = dyn_cast<MemoryEffectOpInterface>(op))
    memory.getEffects(effects);
  return effects;
}

static Value getLocation(Operation *op) {
  for (const MemoryEffects::EffectInstance &effect : getEffects(op))
    if (Value value = effect.getValue())
      return value;
  return op->getNumOperands() ? op->getOperand(0) : Value();
}

static bool isWrite(Operation *op) {
  return llvm::any_of(getEffects(op), [](const auto &effect) {
    return isa<MemoryEffects::Write>(effect.getEffect());
  });
}

static bool hasHazard(Operation *prior, Operation *current,
                      AliasAnalysis &aliasAnalysis) {
  if (!isWrite(prior) && !isWrite(current))
    return false;
  Value lhs = getLocation(prior);
  Value rhs = getLocation(current);
  if (!lhs || !rhs)
    return true;
  return !aliasAnalysis.alias(lhs, rhs).isNo();
}

struct InferMemoryTokens final
    : inter::impl::InferMemoryTokensBase<InferMemoryTokens> {
  void runOnOperation() override {
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      AliasAnalysis aliasAnalysis(function);
      DataFlowConfig config;
      config.setInterprocedural(false);
      DataFlowSolver solver(config);
      solver.load<dataflow::DeadCodeAnalysis>();
      solver.load<inter::MemoryFrontierAnalysis>(aliasAnalysis);
      if (failed(solver.initializeAndRun(function))) {
        function.emitOpError("memory-frontier dataflow failed to converge");
        return signalPassFailure();
      }

      SmallVector<std::pair<Operation *, SmallVector<Value>>> plans;
      DominanceInfo dominance(function);
      WalkResult collected = function.walk([&](Operation *op) {
        if (!isXWMemoryOperation(op) || op->hasAttr("xw.tokens_inferred"))
          return WalkResult::advance();
        const inter::MemoryFrontier *frontier =
            solver.lookupState<inter::MemoryFrontier>(
                solver.getProgramPointBefore(op));
        if (!frontier)
          return WalkResult::advance();
        llvm::SmallPtrSet<Value, 8> seen;
        for (Operation *prior : frontier->getAccesses()) {
          if (prior == op || !isXWMemoryOperation(prior) ||
              !hasHazard(prior, op, aliasAnalysis) || !prior->getNumResults())
            continue;
          Value token = prior->getResult(prior->getNumResults() - 1);
          if (!dominance.dominates(token, op)) {
            if (auto loop = op->getParentOfType<LoopLikeOpInterface>()) {
              (void)loop.getLoopRegions();
              (void)loop.getLoopResults();
            }
            op->emitOpError(
                "requires a memory token from a sibling structured path; "
                "the enclosing RegionBranchOpInterface must expose an "
                "additional yielded token");
            return WalkResult::interrupt();
          }
          if (seen.insert(token).second)
            if (plans.empty() || plans.back().first != op)
              plans.emplace_back(op, SmallVector<Value>());
            plans.back().second.push_back(token);
        }
        return WalkResult::advance();
      });
      if (collected.wasInterrupted())
        return signalPassFailure();

      DenseMap<Value, Value> replacements;
      for (auto &[operation, inferred] : plans) {
        OpBuilder builder(operation);
        SmallVector<Value> allDependencies;
        for (Value operand : operation->getOperands())
          if (isa<xw::MemTokenType>(operand.getType()))
            allDependencies.push_back(operand);
        for (Value token : inferred) {
          Value replacement = replacements.lookup(token);
          allDependencies.push_back(replacement ? replacement : token);
        }

        Value dependency = allDependencies.front();
        if (allDependencies.size() > 1) {
          OperationState joinState(operation->getLoc(), "xw.join");
          joinState.addOperands(allDependencies);
          joinState.addTypes(dependency.getType());
          dependency = builder.create(joinState)->getResult(0);
        }

        OperationState state(operation->getLoc(), operation->getName());
        for (Value operand : operation->getOperands())
          if (!isa<xw::MemTokenType>(operand.getType()))
            state.addOperands(operand);
        state.addOperands(dependency);
        state.addTypes(operation->getResultTypes());
        state.addAttributes(operation->getAttrs());
        state.addAttribute("xw.tokens_inferred", builder.getUnitAttr());
        state.addSuccessors(operation->getSuccessors());
        for (unsigned i = 0; i < operation->getNumRegions(); ++i)
          state.addRegion();
        Operation *replacement = builder.create(state);
        for (auto [oldRegion, newRegion] :
             llvm::zip(operation->getRegions(), replacement->getRegions()))
          newRegion.takeBody(oldRegion);
        for (auto [oldResult, newResult] :
             llvm::zip(operation->getResults(), replacement->getResults()))
          replacements[oldResult] = newResult;
        operation->replaceAllUsesWith(replacement);
        operation->erase();
      }
    }
  }
};

} // namespace.
