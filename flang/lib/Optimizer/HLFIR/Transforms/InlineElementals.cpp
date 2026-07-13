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
/// reads from. Use MemoryEffectOpInterface for a fail-safe implementation.
static mlir::LogicalResult
getReadDependencies(hlfir::ElementalOp elemental,
                    llvm::SmallVectorImpl<mlir::Value> &deps) {
  llvm::SmallPtrSet<mlir::Value, 8> seen;

  mlir::WalkResult walkResult =
      elemental.getRegion().walk([&](mlir::Operation *op) {
        if (mlir::isMemoryEffectFree(op))
          return mlir::WalkResult::advance();

        if (auto memInterface =
                mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
          llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 4> effects;
          memInterface.getEffects(effects);
          bool hasUnspecifiedRead = false;

          for (const auto &effect : effects) {
            if (mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect())) {
              if (mlir::Value val = effect.getValue()) {
                if (seen.insert(val).second)
                  deps.push_back(val);
              } else {
                // Read effect on an unspecified resource (e.g., global state).
                hasUnspecifiedRead = true;
              }
            }
          }

          // If the op has a read effect but the specific value is unknown,
          // conservatively capture all potential reference operands.
          if (hasUnspecifiedRead) {
            // If there are no operands to track, we can't reason about
            // the dependency.
            if (op->getNumOperands() == 0)
              return mlir::WalkResult::interrupt();
            for (mlir::Value operand : op->getOperands()) {
              if (operand.getParentRegion() != &elemental.getRegion()) {
                if (mlir::isa<fir::ReferenceType, fir::PointerType,
                              fir::HeapType, fir::BoxType>(operand.getType())) {
                  if (seen.insert(operand).second)
                    deps.push_back(operand);
                }
              }
            }
          }
          return mlir::WalkResult::advance();
        }

        // Fail-safe: For operations without the interface, conservatively
        // assume we cannot reason about the dependency.
        return mlir::WalkResult::interrupt();
      });

  return mlir::success(!walkResult.wasInterrupted());
}

/// Checks if an operation 'op' potentially modifies any memory location that
/// the elemental reads from (captured in 'deps').
static bool isConflictingWrite(mlir::Operation *op,
                               const llvm::SmallVectorImpl<mlir::Value> &deps,
                               mlir::AliasAnalysis &aa) {
  // Use walk to handle nested regions (fir.if, fir.do_loop, etc.) recursively.
  mlir::WalkResult result = op->walk([&](mlir::Operation *nestedOp) {
    // Operations explicitly marked as having no memory effects are safe.
    if (mlir::isMemoryEffectFree(nestedOp))
      return mlir::WalkResult::advance();

    // Explicitly allow safe HLFIR/FIR metadata/lifetime operations.
    if (mlir::isa<hlfir::DeclareOp, hlfir::AssociateOp, hlfir::EndAssociateOp,
                  fir::AllocaOp, hlfir::NoReassocOp>(nestedOp))
      return mlir::WalkResult::advance();

    // Check for explicit memory effects via the interface.
    if (auto memInterface =
            mlir::dyn_cast<mlir::MemoryEffectOpInterface>(nestedOp)) {
      llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 4> effects;
      memInterface.getEffects(effects);

      for (const auto &effect : effects) {
        // Analyze effects that modify memory or release resources.
        if (mlir::isa<mlir::MemoryEffects::Write, mlir::MemoryEffects::Free>(
                effect.getEffect())) {
          mlir::Value accessedValue = effect.getValue();
          // Fail-safe: Assuming conflict for Unknown resource (e.g. external
          // call).
          if (!accessedValue)
            return mlir::WalkResult::interrupt();

          // Perform alias analysis against all read dependencies.
          for (mlir::Value dep : deps) {
            if (!aa.alias(accessedValue, dep).isNo())
              return mlir::WalkResult::interrupt();
          }
        }
      }
    } else if (nestedOp->getNumRegions() == 0) {
      // Conservative Fallback: If an operation doesn't have interface and
      // has no regions (e.g. a fir.call), assume it can modify anything.
      return mlir::WalkResult::interrupt();
    }

    return mlir::WalkResult::advance();
  });

  // Conflict found as walk interrupted.
  return result.wasInterrupted();
}

static bool isSafeToInline(hlfir::ElementalOp producer,
                           hlfir::ApplyOp applySite, mlir::AliasAnalysis &aa,
                           mlir::DominanceInfo &domInfo) {
  if (!domInfo.properlyDominates(producer.getOperation(),
                                 applySite.getOperation()))
    return false;

  llvm::SmallVector<mlir::Value> deps;
  if (mlir::failed(getReadDependencies(producer, deps)))
    return false;

  mlir::Operation *func = producer->getParentOfType<mlir::func::FuncOp>();
  if (!func)
    return false;

  // Check for conflicting writes between the producer and the apply site.
  mlir::WalkResult result = func->walk([&](mlir::Operation *op) {
    if (op == producer.getOperation() || op == applySite.getOperation())
      return mlir::WalkResult::advance();

    // Analyze operations in the execution path from producer to applySite.
    if (domInfo.properlyDominates(producer.getOperation(), op) &&
        domInfo.dominates(op, applySite.getOperation())) {
      // If 'op' contains the applySite (like a loop shell), skip it to avoid
      // false positives. Its internal operations will be visited individually.
      if (op->getBlock() == applySite.getOperation()->getBlock()) {
        if (isConflictingWrite(op, deps, aa))
          return mlir::WalkResult::interrupt();
      } else if (!op->isAncestor(applySite.getOperation())) {
        // Check operations in sibling blocks or preceding control-flow paths.
        if (isConflictingWrite(op, deps, aa))
          return mlir::WalkResult::interrupt();
      }
    }
    return mlir::WalkResult::advance();
  });

  return !result.wasInterrupted();
}

/// Traces the elemental's dataflow to find its unique apply and destroy
/// operations. Returns the destroy op only if no other consumers require the
/// array buffer.
static std::optional<std::pair<hlfir::ApplyOp, hlfir::DestroyOp>>
getTwoUses(hlfir::ElementalOp elemental, mlir::AliasAnalysis &aliasAnalysis,
           mlir::DominanceInfo &domInfo) {
  // If the ElementalOp must produce a temporary (e.g. for
  // finalization purposes), then we cannot inline it.
  if (hlfir::elementalOpMustProduceTemp(elemental))
    return std::nullopt;

  hlfir::ApplyOp apply;
  hlfir::DestroyOp destroy;
  unsigned applyCount = 0;
  bool hasOtherUsers = false;

  llvm::SmallVector<mlir::Value> worklist;
  worklist.push_back(elemental.getResult());
  llvm::SmallPtrSet<mlir::Value, 16> visited;
  llvm::SmallPtrSet<mlir::Operation *, 4> uniqueApplies;

  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!current || !visited.insert(current).second)
      continue;

    for (mlir::OpOperand &use : current.getUses()) {
      mlir::Operation *user = use.getOwner();

      mlir::TypeSwitch<mlir::Operation *, void>(user)
          .Case<hlfir::ApplyOp>([&](hlfir::ApplyOp op) {
            // Use raw operation pointer to ensure each apply site is
            // counted only once.
            if (uniqueApplies.insert(op.getOperation()).second) {
              apply = op;
              applyCount++;
            }
          })
          .Case<hlfir::DestroyOp>([&](hlfir::DestroyOp op) {
            // Track the mandatory destroy operation for the elemental expr.
            destroy = op;
          })
          .Case<hlfir::DeclareOp, fir::ConvertOp>([&](mlir::Operation *op) {
            // Follow the dataflow through all results of the operation.
            // For hlfir.declare, this catches both the variable and base
            // results. For fir.convert, this catches the converted result.
            for (mlir::Value result : op->getResults()) {
              worklist.push_back(result);
            }
          })
          // Buffer Consumers - These require the destroy to stay.
          .Case<hlfir::AssociateOp, hlfir::SumOp, hlfir::AssignOp,
                hlfir::DesignateOp, fir::CallOp, fir::StoreOp, fir::BoxAddrOp>(
              [&](mlir::Operation *) { hasOtherUsers = true; })
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
            // Only forward if the parent is an op that yields values out.
            if (parent &&
                mlir::isa<mlir::RegionBranchOpInterface, fir::IfOp,
                          fir::DoLoopOp, hlfir::ElementalOp>(parent)) {
              for (auto it : llvm::enumerate(op.getOperands())) {
                if (it.value() == current) {
                  // Map the result index to the parent's result index.
                  unsigned i = it.index();
                  if (i < parent->getNumResults()) {
                    worklist.push_back(parent->getResult(i));
                  }
                }
              }
            } else {
              // If it's a terminator for an unknown op.
              hasOtherUsers = true;
            }
          })
          .Default([&](mlir::Operation *op) {
            if (op->getNumRegions() > 0) {
              // Follow the value through metadata ops (declare, convert, etc.)
              // nested inside regions.
              op->walk([&](mlir::Operation *innerOp) {
                for (mlir::Value operand : innerOp->getOperands()) {
                  if (operand == current) {
                    if (auto nestedApply =
                            mlir::dyn_cast<hlfir::ApplyOp>(innerOp)) {
                      // Use a set to prevent double-counting if walker
                      // and worklist hit the same apply site.
                      if (uniqueApplies.insert(nestedApply.getOperation())
                              .second) {
                        apply = nestedApply;
                        applyCount++;
                      }
                    } else if (mlir::isa<hlfir::DeclareOp, fir::ConvertOp>(
                                   innerOp)) {
                      // Feed internal metadata results back into the worklist.
                      for (mlir::Value res : innerOp->getResults())
                        worklist.push_back(res);
                    } else if (mlir::isa<hlfir::DestroyOp, fir::ResultOp,
                                         mlir::BranchOpInterface>(innerOp)) {
                      // Known safe - control flow and cleanup.
                    } else {
                      // If it's an intrinsic, calls or unknown consumer,
                      // it needs the buffer.
                      hasOtherUsers = true;
                    }
                  }
                }
              });
            } else {
              // Non-region op not handled by specific Case<> (e.g. hlfir.sum)
              hasOtherUsers = true;
            }
          });
      if (applyCount > 1)
        return std::nullopt;
    }
  }

  // Only inline if there is a unique 'apply' site. Other users (such as
  // intrinsic operations) are allowed because scalarizing the elemental
  // renders the original array result redundant.
  if (applyCount != 1 || !destroy)
    return std::nullopt;

  // Verify memory effect and dataflow analysis.
  if (!isSafeToInline(elemental, apply, aliasAnalysis, domInfo))
    return std::nullopt;

  // we can't inline if the return type of the yield doesn't match the return
  // type of the apply
  auto yield = mlir::dyn_cast_or_null<hlfir::YieldElementOp>(
      elemental.getRegion().back().back());
  assert(yield && "hlfir.elemental should always end with a yield");
  if (apply.getResult().getType() != yield.getElementValue().getType())
    return std::nullopt;

  // Only return the destroy op if there's exactly one apply and no other users.
  bool safeToDelete = (applyCount == 1 && !hasOtherUsers);
  return std::make_pair(apply, safeToDelete ? destroy : nullptr);
}

namespace {
class InlineElementalConversion
    : public mlir::OpRewritePattern<hlfir::ElementalOp> {
public:
  using mlir::OpRewritePattern<hlfir::ElementalOp>::OpRewritePattern;
  explicit InlineElementalConversion(mlir::MLIRContext *context,
                                     mlir::AliasAnalysis &aa,
                                     mlir::DominanceInfo &di)
      : OpRewritePattern<hlfir::ElementalOp>(context), aliasAnalysis(aa),
        domInfo(di) {}
  llvm::LogicalResult
  matchAndRewrite(hlfir::ElementalOp elemental,
                  mlir::PatternRewriter &rewriter) const override {
    std::optional<std::pair<hlfir::ApplyOp, hlfir::DestroyOp>> maybeTuple =
        getTwoUses(elemental, aliasAnalysis, domInfo);
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
    // Only erase the destroy and elemental if the analysis shows it's safe.
    if (hlfir::DestroyOp destroyOp = maybeTuple->second) {
      // IR has no users left.
      if (destroyOp->use_empty())
        rewriter.eraseOp(destroyOp);

      if (elemental.getResult().use_empty())
        rewriter.eraseOp(elemental);
    }
    return mlir::success();
  }

private:
  mlir::AliasAnalysis &aliasAnalysis;
  mlir::DominanceInfo &domInfo;
};

class InlineElementalsPass
    : public hlfir::impl::InlineElementalsBase<InlineElementalsPass> {
public:
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();

    // Get AliasAnalysis from the pass manager.
    mlir::AliasAnalysis &aliasAnalysis = getAnalysis<mlir::AliasAnalysis>();
    mlir::DominanceInfo &domInfo = getAnalysis<mlir::DominanceInfo>();
    mlir::GreedyRewriteConfig config;
    // Prevent the pattern driver from merging blocks.
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);

    mlir::RewritePatternSet patterns(context);
    patterns.insert<InlineElementalConversion>(context, aliasAnalysis, domInfo);

    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      mlir::emitError(getOperation()->getLoc(),
                      "failure in HLFIR elemental inlining");
      signalPassFailure();
    }
  }
};
} // namespace
