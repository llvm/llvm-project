#include "inter/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace inter {
#define GEN_PASS_DEF_PREPARECOUNTEDLOOPS
#include "inter/Transforms/Passes.h.inc"
} // namespace inter

using namespace mlir;

namespace {

static std::optional<arith::CmpIPredicate>
convertPredicate(LLVM::ICmpPredicate predicate) {
  switch (predicate) {
  case LLVM::ICmpPredicate::slt:
    return arith::CmpIPredicate::slt;
  case LLVM::ICmpPredicate::sgt:
    return arith::CmpIPredicate::sgt;
  default:
    return std::nullopt;
  }
}

static arith::IntegerOverflowFlags
convertOverflowFlags(LLVM::IntegerOverflowFlags flags) {
  arith::IntegerOverflowFlags converted = arith::IntegerOverflowFlags::none;
  if (LLVM::bitEnumContainsAny(flags, LLVM::IntegerOverflowFlags::nsw))
    converted = converted | arith::IntegerOverflowFlags::nsw;
  if (LLVM::bitEnumContainsAny(flags, LLVM::IntegerOverflowFlags::nuw))
    converted = converted | arith::IntegerOverflowFlags::nuw;
  return converted;
}

struct PrepareCountedLoops final
    : inter::impl::PrepareCountedLoopsBase<PrepareCountedLoops> {
  void runOnOperation() override {
    getOperation().walk([](scf::WhileOp loop) {
      Block *before = loop.getBeforeBody();
      if (!llvm::hasSingleElement(before->without_terminator()))
        return;

      LLVM::ICmpOp compare = dyn_cast<LLVM::ICmpOp>(before->front());
      if (!compare || !compare->hasOneUse() ||
          loop.getConditionOp().getCondition() != compare.getResult())
        return;
      std::optional<arith::CmpIPredicate> predicate =
          convertPredicate(compare.getPredicate());
      if (!predicate)
        return;

      scf::YieldOp yield = loop.getYieldOp();
      for (OpOperand &operand : yield->getOpOperands()) {
        LLVM::AddOp add = operand.get().getDefiningOp<LLVM::AddOp>();
        if (!add)
          continue;

        OpBuilder builder(add);
        arith::AddIOp replacement = arith::AddIOp::create(
            builder, add.getLoc(), add.getLhs(), add.getRhs());
        replacement.setOverflowFlags(
            convertOverflowFlags(add.getOverflowFlags()));
        operand.set(replacement);
        if (add->use_empty())
          add.erase();
      }

      OpBuilder builder(compare);
      arith::CmpIOp replacement =
          arith::CmpIOp::create(builder, compare.getLoc(), *predicate,
                                compare.getLhs(), compare.getRhs());
      compare.getResult().replaceAllUsesWith(replacement);
      compare.erase();
    });

    RewritePatternSet patterns(&getContext());
    scf::populateUpliftWhileToForPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
