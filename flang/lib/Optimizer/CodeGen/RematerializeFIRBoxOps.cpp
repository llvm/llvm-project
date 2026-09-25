//===-- RematerializeFIRBoxOps.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Some NoMemoryEffect FIR operations create allocas as an implementation detail
// of their conversion to the LLVM dialect. These allocas must not be
// accidentally shared across different threads when OpenMP outlining is used.
// This pass rematerializes selected operations into the outlined regions,
// which ensures that the allocas are correctly located inside of the outlined
// function.
//
// Operations rematerialized by this pass are re-created at each use inside of
// the affected regions. LLVM-IR CSE later in the pipeline should merge these
// where possible. However, CSE must not be run between this pass and the
// FIR-to-LLVM conversion because MLIR CSE will completely undo the actions of
// this pass. This is because the side effects on the FIR operations do not
// represent the side effects produced by their implementation in the LLVM
// dialect. This pass makes FIR-to-LLVM descriptor allocation safe for outlined
// regions.

#include "flang/Optimizer/CodeGen/CodeGen.h"

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace fir {
#define GEN_PASS_DEF_REMATERIALIZEFIRBOXOPSPASS
#include "flang/Optimizer/CodeGen/CGPasses.h.inc"
} // namespace fir

namespace {

/// Returns true if \p op has a region that should be rematerialized into.
static bool isRematerializationRegionOp(mlir::Operation *op) {
  return mlir::isa<mlir::omp::OutlineableOpenMPOpInterface>(op);
}

/// Return true for FIR box/class conversions whose operand may need to be
/// rematerialized while preserving the type expected by the use.
static bool isBoxConvert(mlir::Operation *op) {
  auto convert = mlir::dyn_cast_or_null<fir::ConvertOp>(op);
  return convert && mlir::isa<fir::BaseBoxType>(convert.getValue().getType()) &&
         mlir::isa<fir::BaseBoxType>(convert.getType());
}

/// Return true if \p op should be cloned into rematerialization regions.
static bool shouldRematerialize(mlir::Operation *op) {
  if (!op)
    return false;

  if (mlir::isa<fir::EmboxOp, fir::ReboxOp>(op)) {
    assert(
        mlir::isMemoryEffectFree(op) &&
        "This transformation is not safe for operations with memory effects");
    // Not all Embox and Rebox operations are speculatable. This should be safe
    // because SSA can only express a use of the non-speculatable value inside
    // of the region requiring rematerialization if that non-speculatable value
    // always dominates the region. Therefore we are not adding any new UB from
    // absent boxes/arguments. Furthermore, the newly rematerialized operations
    // are created at the use site of the original value as a further guarantee
    // that the rematerializations are only executed if the original value was
    // executed.
    return true;
  }

  // Rematerializing box-to-box conversions is safe and allows rematerialization
  // of emboxes which are only used inside of the region through box-to-box
  // conversions. Conversions to !fir.box<none> are common before calls to
  // runtime functions.
  if (isBoxConvert(op)) {
    assert(
        mlir::isPure(op) &&
        "This transformation is not safe for operations with memory effects");
    return shouldRematerialize(
        mlir::cast<fir::ConvertOp>(op).getValue().getDefiningOp());
  }

  return false;
}

/// Return true if \p definingOp can be rematerialized into \p useRegion.
/// The use can be rematerialized if the defining operation is located
/// in an ancestor region of the use and the definition operation is
/// rematerializable.
static bool canRematerializeDefInRegion(mlir::Operation *definingOp,
                                        mlir::Region &useRegion) {
  if (!definingOp || !shouldRematerialize(definingOp))
    return false;

  mlir::Region *opRegion = definingOp->getParentRegion();
  for (mlir::Region *ancestor = useRegion.getParentRegion(); ancestor;
       ancestor = ancestor->getParentRegion()) {
    if (opRegion == ancestor)
      return true;
  }
  return false;
}

static mlir::Value cloneRematerializedValue(
    mlir::IRRewriter &rewriter, mlir::Region &useRegion, mlir::Value value,
    mlir::IRMapping &mapping,
    llvm::SmallVectorImpl<mlir::Operation *> &eraseCandidates) {
  if (mlir::Value mappedValue = mapping.lookupOrNull(value))
    return mappedValue;

  mlir::Operation *definingOp = value.getDefiningOp();
  if (!canRematerializeDefInRegion(definingOp, useRegion))
    return value;

  // Clone rematerializable dependencies first so the cloned operation uses
  // cloned operands when possible.
  for (mlir::Value operand : definingOp->getOperands())
    cloneRematerializedValue(rewriter, useRegion, operand, mapping,
                             eraseCandidates);

  rewriter.clone(*definingOp, mapping);
  eraseCandidates.push_back(definingOp);
  return mapping.lookup(value);
}

struct RematerializationSite {
  /// Operation whose operands will be rewritten. This is the insertion point
  /// for the rematerialized operations.
  mlir::Operation *user;
  /// Operands of \c user that use values needing rematerialization.
  llvm::SmallVector<mlir::OpOperand *> uses;

  RematerializationSite(mlir::Operation *user,
                        llvm::SmallVector<mlir::OpOperand *> uses)
      : user(user), uses(std::move(uses)) {}
};

/// Walk \p regionOwner and collect all rematerializable operands that use
/// values defined outside the region.
static void collectRematerializableUses(
    mlir::Operation *regionOwner,
    llvm::SmallVectorImpl<RematerializationSite> &rematSites) {
  regionOwner->walk<mlir::WalkOrder::PreOrder>(
      [&](mlir::Operation *op) -> mlir::WalkResult {
        // Don't walk into nested rematerialization regions. They will be
        // processed in their own calls to this function.
        if (op != regionOwner && isRematerializationRegionOp(op))
          return mlir::WalkResult::skip();

        llvm::SmallVector<mlir::OpOperand *> rematerializableUses;
        for (mlir::OpOperand &operand : op->getOpOperands()) {
          if (canRematerializeDefInRegion(operand.get().getDefiningOp(),
                                          *op->getParentRegion()))
            rematerializableUses.push_back(&operand);
        }
        if (!rematerializableUses.empty())
          rematSites.emplace_back(op, std::move(rematerializableUses));

        return mlir::WalkResult::advance();
      });
}

/// Rematerialize supported values defined outside of \p regionOwner into the
/// region
static void rematerializeInRegion(mlir::IRRewriter &rewriter,
                                  mlir::Operation *regionOwner) {
  llvm::SmallVector<RematerializationSite> rematSites;
  collectRematerializableUses(regionOwner, rematSites);
  if (rematSites.empty())
    return;

  mlir::OpBuilder::InsertionGuard guard(rewriter);

  llvm::SmallVector<mlir::Operation *> eraseCandidates;
  for (RematerializationSite &rematSite : rematSites) {
    rewriter.setInsertionPoint(rematSite.user);
    mlir::IRMapping mapping;
    for (mlir::OpOperand *use : rematSite.uses) {
      mlir::Region *useRegion = use->getOwner()->getParentRegion();
      mlir::Value newValue = cloneRematerializedValue(
          rewriter, *useRegion, use->get(), mapping, eraseCandidates);
      use->set(newValue);
    }
  }

  llvm::DenseSet<mlir::Operation *> erased;
  for (mlir::Operation *op : llvm::reverse(eraseCandidates))
    if (erased.insert(op).second && op->use_empty())
      rewriter.eraseOp(op);
}

class RematerializeFIRBoxOpsPass
    : public fir::impl::RematerializeFIRBoxOpsPassBase<
          RematerializeFIRBoxOpsPass> {
public:
  using RematerializeFIRBoxOpsPassBase<
      RematerializeFIRBoxOpsPass>::RematerializeFIRBoxOpsPassBase;

  void runOnOperation() override final {
    mlir::Operation *top = getOperation();

    llvm::SmallVector<mlir::Operation *> regionOwners;
    top->walk([&](mlir::Operation *op) {
      if (isRematerializationRegionOp(op))
        regionOwners.push_back(op);
    });

    mlir::IRRewriter rewriter(top->getContext());
    for (mlir::Operation *op : regionOwners)
      rematerializeInRegion(rewriter, op);
  }
};

} // namespace
