//===- InlineElementals.cpp - Inline chained hlfir.elemental ops ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Chained elemental operations like a + b + c can inline the first elemental
// at the hlfir.apply in the body of the second one (as described in
// docs/HighLevelFIR.md). This has to be done in a pass rather than in lowering
// so that it happens after the HLFIR intrinsic simplification pass.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include <iterator>

namespace hlfir {
#define GEN_PASS_DEF_INLINEELEMENTALS
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

/// Collects all memory values (buffers/references) that the elemental body
/// reads from.
static void getReadDependencies(hlfir::ElementalOp elemental,
                                llvm::SmallVectorImpl<mlir::Value> &deps) {
  elemental.getRegion().walk([&](mlir::Operation *op) {
    if (auto designate = mlir::dyn_cast<hlfir::DesignateOp>(op))
      deps.push_back(designate.getMemref());
    else if (auto load = mlir::dyn_cast<fir::LoadOp>(op))
      deps.push_back(load.getMemref());
    // Capture any value defined outside the elemental but used inside it.
    for (mlir::Value operand : op->getOperands()) {
      if (operand.getParentRegion() != &elemental.getRegion())
        if (mlir::isa<fir::ReferenceType, fir::PointerType, fir::HeapType,
                      fir::BoxType>(operand.getType()))
          deps.push_back(operand);
    }
  });
}

/// Checks if an operation 'op' potentially modifies any memory location that
/// the elemental reads from (captured in 'deps').
static bool isConflictingWrite(mlir::Operation *op,
                               const llvm::SmallVectorImpl<mlir::Value> &deps,
                               mlir::AliasAnalysis &aa) {
  // Operations explicitly marked as having no memory effects are safe.
  if (mlir::isMemoryEffectFree(op))
    return false;

  // Explicitly allow safe HLFIR/FIR metadata/lifetime operations.
  // While these may have internal effects (e.g. allocating a descriptor),
  // they do not modify the user data being read by the elemental.
  if (mlir::isa<hlfir::DeclareOp, hlfir::AssociateOp, hlfir::EndAssociateOp,
                fir::AllocaOp, hlfir::NoReassocOp>(op))
    return false;

  // Check for explicit memory effects via the MemoryEffectOpInterface.
  if (auto memInterface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 4> effects;
    memInterface.getEffects(effects);

    for (const auto &effect : effects) {
      // Analyze effects that modify memory or release resources.
      if (mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect()) ||
          mlir::isa<mlir::MemoryEffects::Free>(effect.getEffect())) {

        mlir::Value accessedValue = effect.getValue();
        // If the effect is on an unknown resource (e.g. external call),
        // assume a conflict.
        if (!accessedValue)
          return true;

        // Perform alias analysis against all read dependencies.
        for (mlir::Value dep : deps) {
          if (!aa.alias(accessedValue, dep).isNo())
            return true;
        }
      }
    }
  } else if (op->getNumRegions() == 0) {
    // Conservative Fallback: If an operation lacks the interface and has no
    // regions (e.g. a fir.call to an external function), assume it can
    // potentially modifies any memory.
    return true;
  }

  // Recursive Analysis into structured control flow regions.
  // (e.g. fir.if, fir.do_loop) to find nested conflicting writes.
  for (mlir::Region &region : op->getRegions()) {
    for (mlir::Block &block : region) {
      for (mlir::Operation &nestedOp : block) {
        if (isConflictingWrite(&nestedOp, deps, aa))
          return true;
      }
    }
  }

  return false;
}

bool isSafeToInline(hlfir::ElementalOp producer, hlfir::ApplyOp applySite,
                    mlir::AliasAnalysis &aa) {
  mlir::DominanceInfo domInfo(producer->getParentOp());
  if (!domInfo.properlyDominates(producer.getOperation(),
                                 applySite.getOperation()))
    return false;

  llvm::SmallVector<mlir::Value> deps;
  getReadDependencies(producer, deps);

  mlir::Operation *func = producer->getParentOfType<mlir::func::FuncOp>();
  bool conflict = false;

  func->walk([&](mlir::Operation *op) {
    // Skip the producer and applySite themselves.
    if (op == producer.getOperation() || op == applySite.getOperation())
      return mlir::WalkResult::advance();

    // Skip the operation that contains the applySite.
    // We only care about operations that execute before the applySite
    // starts or between the producer and the start of the loop.
    if (op->isAncestor(applySite.getOperation()))
      return mlir::WalkResult::advance();

    // Only check operations that strictly execute between definition and use.
    if (domInfo.properlyDominates(producer.getOperation(), op) &&
        domInfo.dominates(op, applySite.getOperation())) {
      if (isConflictingWrite(op, deps, aa)) {
        conflict = true;
        return mlir::WalkResult::interrupt();
      }
    }
    return mlir::WalkResult::advance();
  });

  return !conflict;
}

/// If the elemental has only two uses and those two are an apply operation and
/// a destroy operation, return those two, otherwise return {}
static std::optional<std::pair<hlfir::ApplyOp, hlfir::DestroyOp>>
getTwoUses(hlfir::ElementalOp elemental, mlir::AliasAnalysis &aliasAnalysis) {
  // If the ElementalOp must produce a temporary (e.g. for
  // finalization purposes), then we cannot inline it.
  if (hlfir::elementalOpMustProduceTemp(elemental))
    return std::nullopt;

  hlfir::ApplyOp apply;
  hlfir::DestroyOp destroy;
  unsigned applyCount = 0;

  llvm::SmallVector<mlir::Value> worklist;
  worklist.push_back(elemental.getResult());
  llvm::SmallPtrSet<mlir::Value, 16> visited;

  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!current || !visited.insert(current).second)
      continue;

    for (mlir::OpOperand &use : current.getUses()) {
      mlir::Operation *user = use.getOwner();

      mlir::TypeSwitch<mlir::Operation *, void>(user)
          .Case<hlfir::ApplyOp>([&](hlfir::ApplyOp op) {
            apply = op;
            applyCount++;
          })
          .Case<hlfir::DestroyOp>([&](hlfir::DestroyOp op) {
            // Track the mandatory destroy operation for the elemental expr.
            destroy = op;
          })
          .Case<hlfir::DeclareOp>([&](hlfir::DeclareOp op) {
            // Follow the dataflow through variable declarations.
            worklist.push_back(op.getBase());
          })
          .Case<fir::ConvertOp>([&](fir::ConvertOp op) {
            // Follow the dataflow through type conversions.
            worklist.push_back(op.getResult());
          })
          .Case<mlir::BranchOpInterface>([&](mlir::BranchOpInterface branch) {
            for (unsigned i = 0; i < branch->getNumSuccessors(); ++i) {
              mlir::SuccessorOperands operands = branch.getSuccessorOperands(i);
              for (unsigned j = 0; j < operands.size(); ++j) {
                if (operands[j] == current) {
                  // The j-th operand of the branch maps to the j-th block
                  // argument of the successor block.
                  mlir::Block *successor = branch->getSuccessor(i);
                  worklist.push_back(successor->getArgument(j));
                }
              }
            }
          })
          .Case<fir::ResultOp>([&](fir::ResultOp op) {
            mlir::Operation *parent = op->getParentOp();
            if (parent) {
              for (auto it : llvm::enumerate(op.getOperands())) {
                if (it.value() == current) {
                  // 'current' is being yielded. The value outside the loop is
                  // the i-th result of the parent operation.
                  unsigned i = it.index();
                  if (i < parent->getNumResults()) {
                    worklist.push_back(parent->getResult(i));
                  }
                }
              }
            }
          })
          .Default([&](mlir::Operation *op) {
            // If the elemental result is used by an operation with regions
            // (like fir.if or fir.do_loop), the apply site may be nested
            // inside.
            if (op->getNumRegions() > 0) {
              op->walk([&](hlfir::ApplyOp nestedApply) {
                if (nestedApply.getExpr() == current) {
                  apply = nestedApply;
                  applyCount++;
                }
              });
            }
          });
    }
  }

  // Only inline if there is a unique 'apply' site. Other users (such as
  // intrinsic operations) are allowed because scalarizing the elemental
  // renders the original array result redundant.
  if (applyCount != 1 || !destroy)
    return std::nullopt;

  // Verify memory effect and dataflow analysis.
  if (!isSafeToInline(elemental, apply, aliasAnalysis))
    return std::nullopt;

  // we can't inline if the return type of the yield doesn't match the return
  // type of the apply
  auto yield = mlir::dyn_cast_or_null<hlfir::YieldElementOp>(
      elemental.getRegion().back().back());
  assert(yield && "hlfir.elemental should always end with a yield");
  if (apply.getResult().getType() != yield.getElementValue().getType())
    return std::nullopt;

  return std::pair{apply, destroy};
}

namespace {
class InlineElementalConversion
    : public mlir::OpRewritePattern<hlfir::ElementalOp> {
public:
  using mlir::OpRewritePattern<hlfir::ElementalOp>::OpRewritePattern;
  explicit InlineElementalConversion(mlir::MLIRContext *context,
                                     mlir::AliasAnalysis &aa)
      : OpRewritePattern<hlfir::ElementalOp>(context), aliasAnalysis(aa) {}
  llvm::LogicalResult
  matchAndRewrite(hlfir::ElementalOp elemental,
                  mlir::PatternRewriter &rewriter) const override {
    std::optional<std::pair<hlfir::ApplyOp, hlfir::DestroyOp>> maybeTuple =
        getTwoUses(elemental, aliasAnalysis);
    if (!maybeTuple)
      return rewriter.notifyMatchFailure(
          elemental, "hlfir.elemental is not a candidate for inlining");

    if (elemental.isOrdered()) {
      // We can only inline the ordered elemental into a loop-like
      // construct that processes the indices in-order and does not
      // have the side effects itself. Adhere to conservative behavior
      // for the time being.
      return rewriter.notifyMatchFailure(elemental,
                                         "hlfir.elemental is ordered");
    }
    auto [apply, destroy] = *maybeTuple;

    assert(elemental.getRegion().hasOneBlock() &&
           "expect elemental region to have one block");

    fir::FirOpBuilder builder{rewriter, elemental.getOperation()};
    builder.setInsertionPointAfter(apply);
    hlfir::YieldElementOp yield = hlfir::inlineElementalOp(
        elemental.getLoc(), builder, elemental, apply.getIndices());

    // remove the old elemental and all of the bookkeeping
    rewriter.replaceOp(apply, {yield.getElementValue()});
    rewriter.eraseOp(yield);
    rewriter.eraseOp(destroy);
    // Only erase the elemental if that was its last use.
    if (elemental->use_empty())
      rewriter.eraseOp(elemental);

    return mlir::success();
  }

private:
  mlir::AliasAnalysis &aliasAnalysis;
};

class InlineElementalsPass
    : public hlfir::impl::InlineElementalsBase<InlineElementalsPass> {
public:
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();

    // Get AliasAnalysis from the pass manager.
    mlir::AliasAnalysis &aliasAnalysis = getAnalysis<mlir::AliasAnalysis>();
    mlir::GreedyRewriteConfig config;
    // Prevent the pattern driver from merging blocks.
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);

    mlir::RewritePatternSet patterns(context);
    patterns.insert<InlineElementalConversion>(context, aliasAnalysis);

    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      mlir::emitError(getOperation()->getLoc(),
                      "failure in HLFIR elemental inlining");
      signalPassFailure();
    }
  }
};
} // namespace
