#include "inter/Analysis/MemoryFrontierAnalysis.h"
#include "inter/Dialect/Inter/IR/XW.h"

#include "inter/Transforms/Passes.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace inter {
#define GEN_PASS_DEF_INFERMEMORYTOKENS
#include "inter/Transforms/Passes.h.inc"
} // namespace inter.

using namespace mlir;

namespace {

static bool isXWMemoryOperation(Operation *op) {
  if (op->getName().getDialectNamespace() !=
          xw::XWDialect::getDialectNamespace() ||
      !isa<MemoryEffectOpInterface>(op))
    return false;
  auto isTokenType = [](Type type) { return isa<xw::MemTokenType>(type); };
  return llvm::any_of(op->getOperandTypes(), isTokenType) ||
         llvm::any_of(op->getResultTypes(), isTokenType);
}

static bool isStructuredOperation(Operation *operation) {
  return isa<scf::IfOp, xw::WhereOp, scf::ForOp, scf::WhileOp>(operation);
}

static bool isToken(Value value) {
  return isa<xw::MemTokenType>(value.getType());
}

static SmallVector<MemoryEffects::EffectInstance> getEffects(Operation *op) {
  SmallVector<MemoryEffects::EffectInstance> effects;
  if (MemoryEffectOpInterface memory = dyn_cast<MemoryEffectOpInterface>(op))
    memory.getEffects(effects);
  return effects;
}

static bool isBarrier(Operation *op) { return isa<xw::BarrierOp>(op); }

static bool isDefaultMemoryEffect(MemoryEffects::EffectInstance effect) {
  return effect.getResource() == SideEffects::DefaultResource::get();
}

static bool hasWrite(Operation *op) {
  return llvm::any_of(getEffects(op), [](MemoryEffects::EffectInstance effect) {
    return isDefaultMemoryEffect(effect) &&
           isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect());
  });
}

static SmallVector<Value> getLocations(Operation *op) {
  SmallVector<Value> locations;
  for (MemoryEffects::EffectInstance effect : getEffects(op))
    if (isDefaultMemoryEffect(effect) &&
        isa<MemoryEffects::Read, MemoryEffects::Write, MemoryEffects::Free>(
            effect.getEffect()))
      if (Value value = effect.getValue())
        locations.push_back(value);
  return locations;
}

static bool hasRead(Operation *op) {
  return llvm::any_of(getEffects(op), [](MemoryEffects::EffectInstance effect) {
    return isDefaultMemoryEffect(effect) &&
           isa<MemoryEffects::Read>(effect.getEffect());
  });
}

static bool mayAlias(Operation *lhs, Operation *rhs,
                     AliasAnalysis &aliasAnalysis) {
  SmallVector<Value> lhsLocations = getLocations(lhs);
  SmallVector<Value> rhsLocations = getLocations(rhs);
  if (lhsLocations.empty() || rhsLocations.empty())
    return true;
  return llvm::any_of(lhsLocations, [&](Value lhsLocation) {
    return llvm::any_of(rhsLocations, [&](Value rhsLocation) {
      return !aliasAnalysis.alias(lhsLocation, rhsLocation).isNo();
    });
  });
}

static bool hasHazard(Operation *prior, Operation *current,
                      AliasAnalysis &aliasAnalysis) {
  if (isBarrier(prior) || isBarrier(current))
    return true;
  if (!hasWrite(prior) && !hasWrite(current))
    return false;
  return mayAlias(prior, current, aliasAnalysis);
}

static void appendPlan(DenseMap<unsigned, SmallVector<unsigned>> &plans,
                       const DenseMap<Operation *, unsigned> &identifiers,
                       Operation *operation, Operation *predecessor) {
  unsigned identifier = identifiers.lookup(operation);
  unsigned predecessorIdentifier = identifiers.lookup(predecessor);
  if (identifier && predecessorIdentifier &&
      !llvm::is_contained(plans[identifier], predecessorIdentifier))
    plans[identifier].push_back(predecessorIdentifier);
}

static void
planPrefetchDependencies(Block &block, ArrayRef<Operation *> incoming,
                         const DenseMap<Operation *, unsigned> &identifiers,
                         DenseMap<unsigned, SmallVector<unsigned>> &plans,
                         AliasAnalysis &aliasAnalysis) {
  SmallVector<Operation *> pending(incoming);
  for (Operation &operation : block.without_terminator()) {
    if (isa<xw::Block2DPrefetchOp>(operation)) {
      pending.push_back(&operation);
      continue;
    }
    if (isXWMemoryOperation(&operation) && hasRead(&operation)) {
      llvm::erase_if(pending, [&](Operation *prefetch) {
        if (!mayAlias(prefetch, &operation, aliasAnalysis))
          return false;
        appendPlan(plans, identifiers, &operation, prefetch);
        return true;
      });
    }
    if (!isStructuredOperation(&operation))
      continue;
    for (Region &region : operation.getRegions())
      if (!region.empty())
        planPrefetchDependencies(region.front(), pending, identifiers, plans,
                                 aliasAnalysis);
  }
}

static void appendUnique(SmallVectorImpl<Value> &values, Value value) {
  if (value && !llvm::is_contained(values, value))
    values.push_back(value);
}

static Value createInitialToken(OpBuilder &builder, Operation *before) {
  builder.setInsertionPoint(before);
  OperationState state(before->getLoc(), "xw.token");
  state.addTypes(xw::MemTokenType::get(builder.getContext()));
  return builder.create(state)->getResult(0);
}

class TokenRewriter {
public:
  TokenRewriter(MLIRContext *context,
                const DenseMap<Operation *, unsigned> &identifiers,
                const DenseMap<unsigned, SmallVector<unsigned>> &plans,
                const DenseSet<Operation *> &structured)
      : builder(context), identifiers(identifiers), plans(plans),
        structured(structured) {}

  void rewriteFunction(func::FuncOp function) {
    if (function.getBody().empty())
      return;
    rewriteBlock(function.getBody().front(), {});
  }

private:
  Value joinValues(Operation *before, ArrayRef<Value> values) {
    SmallVector<Value> unique;
    for (Value value : values)
      appendUnique(unique, value);
    if (unique.empty())
      return Value();
    if (unique.size() == 1)
      return unique.front();
    builder.setInsertionPoint(before);
    OperationState state(before->getLoc(), "xw.join");
    state.addOperands(unique);
    state.addTypes(xw::MemTokenType::get(builder.getContext()));
    return builder.create(state)->getResult(0);
  }

  SmallVector<Value> rewriteBlock(Block &block, SmallVector<Value> frontier) {
    Value regionIncoming;
    SmallVector<Value> deferredPrefetches;
    if (!isa<func::FuncOp>(block.getParentOp()) && frontier.size() == 1) {
      regionIncoming = frontier.front();
      frontier.clear();
    }
    SmallVector<Operation *> operations;
    for (Operation &operation : block.without_terminator())
      operations.push_back(&operation);
    for (Operation *operation : operations) {
      if (isXWMemoryOperation(operation)) {
        bool barrier = isBarrier(operation);
        SmallVector<Value> incoming;
        appendUnique(incoming, regionIncoming);
        for (Value value : frontier)
          if (barrier || structuredTokens.contains(value))
            appendUnique(incoming, value);
        Value token = rewriteMemory(operation, incoming);
        if (barrier)
          frontier.clear();
        if (isa<xw::Block2DPrefetchOp>(operation))
          appendUnique(deferredPrefetches, token);
        else
          frontier.push_back(token);
        continue;
      }
      if (!structured.contains(operation))
        continue;
      if (operation->hasAttr("xw.tokens_inferred")) {
        for (Region &region : operation->getRegions())
          if (!region.empty())
            rewriteBlock(region.front(), {});
        Value outgoing = operation->getResult(operation->getNumResults() - 1);
        tokens[identifiers.lookup(operation)] = outgoing;
        frontier.clear();
        frontier.push_back(outgoing);
        continue;
      }
      SmallVector<Value> boundaryFrontier = frontier;
      appendUnique(boundaryFrontier, regionIncoming);
      Value incoming = joinValues(operation, boundaryFrontier);
      if (!incoming)
        incoming = createInitialToken(builder, operation);
      Value outgoing = rewriteStructured(operation, incoming);
      tokens[identifiers.lookup(operation)] = outgoing;
      structuredTokens.insert(outgoing);
      frontier.clear();
      frontier.push_back(outgoing);
    }
    for (Value prefetch : deferredPrefetches)
      appendUnique(frontier, prefetch);
    return frontier;
  }

  Value joinDependencies(Operation *operation, ArrayRef<Value> incoming) {
    SmallVector<Value> dependencies;
    for (Value operand : operation->getOperands())
      if (isToken(operand))
        appendUnique(dependencies, operand);
    unsigned identifier = identifiers.lookup(operation);
    for (unsigned predecessor : plans.lookup(identifier))
      appendUnique(dependencies, tokens.lookup(predecessor));
    for (Value value : incoming)
      appendUnique(dependencies, value);
    return joinValues(operation, dependencies);
  }

  Value rewriteMemory(Operation *operation, ArrayRef<Value> incoming) {
    unsigned identifier = identifiers.lookup(operation);
    Value dependency = joinDependencies(operation, incoming);
    if (!dependency && isa<xw::AllocReleaseOp>(operation))
      dependency = createInitialToken(builder, operation);
    builder.setInsertionPoint(operation);
    OperationState state(operation->getLoc(), operation->getName());
    for (Value operand : operation->getOperands())
      if (!isToken(operand))
        state.addOperands(operand);
    if (dependency)
      state.addOperands(dependency);
    state.addTypes(operation->getResultTypes());
    state.addAttributes(operation->getAttrs());
    state.propertiesAttr = operation->getPropertiesAsAttribute();
    state.addAttribute("xw.tokens_inferred", builder.getUnitAttr());
    Operation *replacement = builder.create(state);
    operation->replaceAllUsesWith(replacement);
    operation->erase();
    Value token = replacement->getResult(replacement->getNumResults() - 1);
    tokens[identifier] = token;
    return token;
  }

  Operation *replaceStructured(Operation *operation, Value incoming,
                               bool addInit) {
    SmallVector<Type> resultTypes(operation->getResultTypes());
    builder.setInsertionPoint(operation);
    OperationState state(operation->getLoc(), operation->getName());
    state.addOperands(operation->getOperands());
    if (addInit)
      state.addOperands(incoming);
    state.addTypes(resultTypes);
    state.addTypes(xw::MemTokenType::get(builder.getContext()));
    state.addAttributes(operation->getAttrs());
    state.propertiesAttr = operation->getPropertiesAsAttribute();
    state.addAttribute("xw.tokens_inferred", builder.getUnitAttr());
    unsigned regionCount = operation->getNumRegions();
    for (unsigned index = 0; index < regionCount; ++index)
      state.addRegion()->takeBody(operation->getRegion(index));
    Operation *replacement = builder.create(state);
    for (auto [oldResult, newResult] : llvm::zip(
             operation->getResults(), replacement->getResults().drop_back()))
      oldResult.replaceAllUsesWith(newResult);
    operation->erase();
    return replacement;
  }

  void appendYield(Block &block, Value token) {
    Operation *terminator = block.getTerminator();
    terminator->insertOperands(terminator->getNumOperands(), token);
  }

  Value rewriteIfLike(Operation *operation, Value incoming) {
    Operation *replacement = replaceStructured(operation, incoming, false);
    for (Region &region : replacement->getRegions()) {
      if (region.empty()) {
        Block *block = builder.createBlock(&region);
        builder.setInsertionPointToEnd(block);
        OperationState yieldState(replacement->getLoc(),
                                  isa<scf::IfOp>(replacement) ? "scf.yield"
                                                              : "xw.yield");
        yieldState.addOperands(incoming);
        builder.create(yieldState);
        continue;
      }
      SmallVector<Value> outgoing = rewriteBlock(region.front(), {incoming});
      Value token = joinValues(region.front().getTerminator(), outgoing);
      appendYield(region.front(), token ? token : incoming);
    }
    return replacement->getResult(replacement->getNumResults() - 1);
  }

  Value rewriteFor(Operation *operation, Value incoming) {
    Operation *replacement = replaceStructured(operation, incoming, true);
    Block &body = replacement->getRegion(0).front();
    Value argument =
        body.addArgument(incoming.getType(), replacement->getLoc());
    SmallVector<Value> outgoing = rewriteBlock(body, {argument});
    Value token = joinValues(body.getTerminator(), outgoing);
    appendYield(body, token ? token : argument);
    return replacement->getResult(replacement->getNumResults() - 1);
  }

  Value rewriteWhile(Operation *operation, Value incoming) {
    Operation *replacement = replaceStructured(operation, incoming, true);
    Block &before = replacement->getRegion(0).front();
    Block &after = replacement->getRegion(1).front();
    Value beforeArgument =
        before.addArgument(incoming.getType(), replacement->getLoc());
    Value afterArgument =
        after.addArgument(incoming.getType(), replacement->getLoc());
    SmallVector<Value> beforeOutgoing = rewriteBlock(before, {beforeArgument});
    Value beforeToken = joinValues(before.getTerminator(), beforeOutgoing);
    appendYield(before, beforeToken ? beforeToken : beforeArgument);
    SmallVector<Value> afterOutgoing = rewriteBlock(after, {afterArgument});
    Value afterToken = joinValues(after.getTerminator(), afterOutgoing);
    appendYield(after, afterToken ? afterToken : afterArgument);
    return replacement->getResult(replacement->getNumResults() - 1);
  }

  Value rewriteStructured(Operation *operation, Value incoming) {
    if (isa<scf::IfOp, xw::WhereOp>(operation))
      return rewriteIfLike(operation, incoming);
    if (isa<scf::ForOp>(operation))
      return rewriteFor(operation, incoming);
    return rewriteWhile(operation, incoming);
  }

  OpBuilder builder;
  const DenseMap<Operation *, unsigned> &identifiers;
  const DenseMap<unsigned, SmallVector<unsigned>> &plans;
  const DenseSet<Operation *> &structured;
  DenseMap<unsigned, Value> tokens;
  DenseSet<Value> structuredTokens;
};

struct InferMemoryTokens final
    : inter::impl::InferMemoryTokensBase<InferMemoryTokens> {
  void runOnOperation() override {
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      if (!function.getBody().hasOneBlock()) {
        function.emitOpError(
            "memory token inference requires a single-block function body");
        return signalPassFailure();
      }
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

      DenseMap<Operation *, unsigned> identifiers;
      DenseSet<Operation *> structured;
      unsigned nextIdentifier = 1;
      WalkResult collected = function.walk([&](Operation *operation) {
        if (!isXWMemoryOperation(operation))
          return WalkResult::advance();
        identifiers.try_emplace(operation, nextIdentifier++);
        for (Operation *parent = operation->getParentOp(); parent != function;
             parent = parent->getParentOp()) {
          if (!isStructuredOperation(parent)) {
            operation->emitOpError(
                "is nested in an unsupported region holder '")
                << parent->getName() << "'";
            return WalkResult::interrupt();
          }
          if (!llvm::all_of(parent->getRegions(), [](Region &region) {
                return region.empty() || region.hasOneBlock();
              })) {
            parent->emitOpError(
                "memory token inference requires single-block regions");
            return WalkResult::interrupt();
          }
          structured.insert(parent);
          identifiers.try_emplace(parent, nextIdentifier++);
        }
        return WalkResult::advance();
      });
      if (collected.wasInterrupted())
        return signalPassFailure();

      auto projectToBlock = [&](Operation *predecessor,
                                Block *block) -> Operation * {
        Operation *projected = predecessor;
        while (projected->getBlock() != block) {
          Operation *parent = projected->getParentOp();
          if (!parent || parent == function || !structured.contains(parent))
            return nullptr;
          projected = parent;
        }
        return projected;
      };

      DenseMap<unsigned, SmallVector<unsigned>> plans;
      for (auto [operation, identifier] : identifiers) {
        if (!isXWMemoryOperation(operation))
          continue;
        const inter::MemoryFrontier *frontier =
            solver.lookupState<inter::MemoryFrontier>(
                solver.getProgramPointBefore(operation));
        if (!frontier)
          continue;
        for (Operation *prior : frontier->getAccesses()) {
          if (prior == operation || !isXWMemoryOperation(prior) ||
              !hasHazard(prior, operation, aliasAnalysis))
            continue;
          Operation *projected = projectToBlock(prior, operation->getBlock());
          if (!projected)
            continue;
          unsigned predecessor = identifiers.lookup(projected);
          if (predecessor &&
              !llvm::is_contained(plans[identifier], predecessor))
            plans[identifier].push_back(predecessor);
        }
      }
      planPrefetchDependencies(function.getBody().front(), {}, identifiers,
                               plans, aliasAnalysis);

      TokenRewriter(function.getContext(), identifiers, plans, structured)
          .rewriteFunction(function);
    }
  }
};

} // namespace.
