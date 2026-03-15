//===- ACCUseDeviceCanonicalizer.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass canonicalizes the use_device clause on a host_data construct such
// that use_device(x) can be lowered to a simple runtime call that takes the
// actual host pointer as argument.
//
// For a use_device operand that is a box type or a reference to a box, the
// pass:
//   1. Extracts the host base address for mapping to a device address using
//      acc.use_device.
//   2. Creates a new boxed descriptor with the device address as the base
//      address for use inside the host_data region.
//
// The pass also removes unused use_device clauses, reducing the number of
// runtime calls.
//
// Supported use_device operand types:
//
//   Scalars:
//     - !fir.ref<i32>, !fir.ref<f64>, etc.
//
//   Arrays:
//     - Explicit shape (no descriptor): !fir.ref<!fir.array<100xi32>>
//     - Adjustable size: !fir.ref<!fir.array<?xi32>>
//     - Assumed shape (handled by hoistBox): !fir.box<!fir.array<?xi32>>
//     - Assumed size: !fir.ref<!fir.array<?xi32>>
//     - Deferred shape (handled by hoistRefToBox):
//         - Allocatable: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
//         - Pointer: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
//     - Subarray specification (handled by hoistBox):
//     !fir.box<!fir.array<?xi32>>
//
//   Not yet supported:
//     - Assumed rank arrays
//     - Composite variables: !fir.ref<!fir.type<...>>
//     - Array elements (device pointer arithmetic in host_data region)
//     - Composite variable members
//     - Fortran common blocks: use_device(/cm_block/)
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/OpenACC/Passes.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include <cassert>

namespace fir::acc {
#define GEN_PASS_DEF_ACCUSEDEVICECANONICALIZER
#include "flang/Optimizer/OpenACC/Passes.h.inc"
} // namespace fir::acc

#define DEBUG_TYPE "acc-use-device-canonicalizer"

using namespace mlir;

namespace {

struct UseDeviceHostDataHoisting : public OpRewritePattern<acc::HostDataOp> {
  using OpRewritePattern<acc::HostDataOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(acc::HostDataOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> usedOperands;
    SmallVector<Value> unusedUseDeviceOperands;
    SmallVector<acc::UseDeviceOp> refToBoxUseDeviceOps;
    SmallVector<acc::UseDeviceOp> boxUseDeviceOps;

    for (Value operand : op.getDataClauseOperands()) {
      if (acc::UseDeviceOp useDeviceOp =
              operand.getDefiningOp<acc::UseDeviceOp>()) {
        if (fir::isBoxAddress(useDeviceOp.getVar().getType())) {
          if (!llvm::hasSingleElement(useDeviceOp->getUsers()))
            refToBoxUseDeviceOps.push_back(useDeviceOp);
        } else if (isa<fir::BoxType>(useDeviceOp.getVar().getType())) {
          if (!llvm::hasSingleElement(useDeviceOp->getUsers()))
            boxUseDeviceOps.push_back(useDeviceOp);
        }

        // host_data is the only user of this use_device operand - mark for
        // removal
        if (llvm::hasSingleElement(useDeviceOp->getUsers()))
          unusedUseDeviceOperands.push_back(useDeviceOp.getResult());
        else
          usedOperands.push_back(useDeviceOp.getResult());
      } else {
        // Operand is not an `acc.use_device` result, keep it as is.
        usedOperands.push_back(operand);
      }
    }

    assert(!usedOperands.empty() && "Host_data operation has no used operands");

    if (!unusedUseDeviceOperands.empty()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "ACCUseDeviceCanonicalizer: Removing "
                 << unusedUseDeviceOperands.size()
                 << " unused use_device operands from host_data operation\n");

      // Update the host_data operation to have only used operands
      rewriter.modifyOpInPlace(op, [&]() {
        op.getDataClauseOperandsMutable().assign(usedOperands);
      });

      // Remove unused use_device operations
      for (Value operand : unusedUseDeviceOperands) {
        acc::UseDeviceOp useDeviceOp =
            operand.getDefiningOp<acc::UseDeviceOp>();
        LLVM_DEBUG(llvm::dbgs() << "ACCUseDeviceCanonicalizer: Erasing: "
                                << *useDeviceOp << "\n");
        rewriter.eraseOp(useDeviceOp);
      }
      return success();
    }

    // Handle references to box types
    bool modified = false;
    for (acc::UseDeviceOp useDeviceOp : refToBoxUseDeviceOps)
      modified |=
          hoistRefToBox(rewriter, useDeviceOp.getResult(), useDeviceOp, op);

    // Handle box types
    for (acc::UseDeviceOp useDeviceOp : boxUseDeviceOps)
      modified |= hoistBox(rewriter, useDeviceOp.getResult(), useDeviceOp, op);

    return modified ? success() : failure();
  }

private:
  /// Collect users of `acc.use_device` operation inside the `acc.host_data`
  /// region that need to be updated with the final replacement value.
  void collectUseDeviceUsersToUpdate(
      acc::UseDeviceOp useDeviceOp, acc::HostDataOp hostDataOp,
      SmallVectorImpl<Operation *> &usersToUpdate) const {
    for (mlir::Operation *user : useDeviceOp->getUsers())
      if (hostDataOp.getRegion().isAncestor(user->getParentRegion()))
        usersToUpdate.push_back(user);
  }

  /// Create new `acc.use_device` operation with the given box address as
  /// operand. Updates the `acc.host_data` operation to use the new
  /// `acc.use_device` result.
  acc::UseDeviceOp createNewUseDeviceOp(PatternRewriter &rewriter,
                                        acc::UseDeviceOp useDeviceOp,
                                        acc::HostDataOp hostDataOp,
                                        fir::BoxAddrOp boxAddr) const {
    // Create use_device on the raw pointer
    acc::UseDeviceOp newUseDeviceOp = acc::UseDeviceOp::create(
        rewriter, useDeviceOp.getLoc(), boxAddr.getType(), boxAddr.getResult(),
        useDeviceOp.getVarTypeAttr(), useDeviceOp.getVarPtrPtr(),
        useDeviceOp.getBounds(), useDeviceOp.getAsyncOperands(),
        useDeviceOp.getAsyncOperandsDeviceTypeAttr(),
        useDeviceOp.getAsyncOnlyAttr(), useDeviceOp.getDataClauseAttr(),
        useDeviceOp.getStructuredAttr(), useDeviceOp.getImplicitAttr(),
        useDeviceOp.getModifiersAttr(), useDeviceOp.getNameAttr(),
        useDeviceOp.getRecipeAttr());

    LLVM_DEBUG(llvm::dbgs() << "Created new hoisted pattern for box access:\n"
                            << "  box_addr: " << *boxAddr << "\n"
                            << "  new use_device: " << *newUseDeviceOp << "\n");

    // Replace the old `acc.use_device` operand in the `acc.host_data` operation
    // with the new one
    rewriter.modifyOpInPlace(hostDataOp, [&]() {
      hostDataOp->replaceUsesOfWith(useDeviceOp.getResult(),
                                    newUseDeviceOp.getResult());
    });

    return newUseDeviceOp;
  }

  /// Canonicalize  use_device operand that is a reference to a box.
  /// Transforms:
  ///   %3 = fir.address_of(@_QFEtgt) : !fir.ref<i32>
  ///   %5 = fir.embox %3 : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
  ///   fir.store %5 to %0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  ///   %9 = acc.use_device varPtr(%0 : !fir.ref<!fir.box<!fir.ptr<i32>>>)
  ///   -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "ptr"}
  ///   acc.host_data dataOperands(%9 : !fir.ref<!fir.box<!fir.ptr<i32>>>) {
  ///     %loaded = fir.load %9 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  ///     %addr = fir.box_addr %loaded : (!fir.box<!fir.ptr<i32>>) ->
  ///     !fir.ptr<i32> %conv = fir.convert %addr : (!fir.ptr<i32>) -> i64
  ///     fir.call @foo(%conv) : (i64) -> ()
  ///     acc.terminator
  ///   }
  /// into:
  ///   %loaded = fir.load %0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  ///   %addr = fir.box_addr %loaded : (!fir.box<!fir.ptr<i32>>) ->
  ///   !fir.ptr<i32>
  ///   %dev_ptr = acc.use_device varPtr(%addr : !fir.ptr<i32>) ->
  ///   !fir.ptr<i32>
  ///   -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "ptr"}
  ///   acc.host_data dataOperands(%dev_ptr : !fir.ref<!fir.box<!fir.ptr<i32>>>)
  ///   {
  ///     %embox = fir.embox %dev_ptr : (!fir.ptr<i32>) ->
  ///     !fir.box<!fir.ptr<i32>> %alloca = fir.alloca !fir.box<!fir.ptr<i32>>
  ///     fir.store %embox to %alloca : !fir.ref<!fir.box<!fir.ptr<i32>>>
  ///     %loaded2 = fir.load %alloca : !fir.ref<!fir.box<!fir.ptr<i32>>>
  ///     %addr2 = fir.box_addr %loaded2 : (!fir.box<!fir.ptr<i32>>) ->
  ///     !fir.ptr<i32> %conv = fir.convert %addr2 : (!fir.ptr<i32>) -> i64
  ///     fir.call @foo(%conv) : (i64) -> ()
  ///     acc.terminator
  ///   }
  bool hoistRefToBox(PatternRewriter &rewriter, Value operand,
                     acc::UseDeviceOp useDeviceOp,
                     acc::HostDataOp hostDataOp) const {

    // Safety check: if the use_device operation is already using a box_addr
    // result, it means it has already been processed, so skip to avoid infinite
    // loop
    if (useDeviceOp.getVar().getDefiningOp<fir::BoxAddrOp>()) {
      LLVM_DEBUG(llvm::dbgs() << "ACCUseDeviceCanonicalizer: Skipping "
                                 "already processed use_device operation\n");
      return false;
    }
    // Get the ModuleOp before we erase useDeviceOp to avoid invalid reference
    ModuleOp mod = useDeviceOp->getParentOfType<ModuleOp>();

    // Collect users of the original `acc.use_device` operation that need to be
    // updated
    SmallVector<Operation *> usersToUpdate;
    collectUseDeviceUsersToUpdate(useDeviceOp, hostDataOp, usersToUpdate);

    rewriter.setInsertionPoint(useDeviceOp);
    // Create a load operation to get the box from the variable
    fir::LoadOp box = fir::LoadOp::create(rewriter, useDeviceOp.getLoc(),
                                          useDeviceOp.getVar());
    // Create a box_addr operation to get the address from the box
    fir::BoxAddrOp boxAddr =
        fir::BoxAddrOp::create(rewriter, useDeviceOp.getLoc(), box);

    acc::UseDeviceOp newUseDeviceOp =
        createNewUseDeviceOp(rewriter, useDeviceOp, hostDataOp, boxAddr);

    LLVM_DEBUG(llvm::dbgs()
               << "Created new hoisted pattern for pointer access:\n"
               << "  load box: " << *box << "\n"
               << "  box_addr: " << *boxAddr << "\n"
               << "  new use_device: " << *newUseDeviceOp << "\n");

    // Set insertion point to the first op inside the host_data region
    rewriter.setInsertionPoint(&hostDataOp.getRegion().front().front());

    // Create a FirOpBuilder from the PatternRewriter using the module we got
    // earlier
    fir::FirOpBuilder builder(rewriter, mod);
    Value newBoxwithDevicePtr = fir::factory::getDescriptorWithNewBaseAddress(
        builder, useDeviceOp.getLoc(), box.getResult(),
        newUseDeviceOp.getResult());

    // Create new memory location and store the newBoxwithDevicePtr into new
    // memory location
    fir::AllocaOp newMemLoc = fir::AllocaOp::create(
        rewriter, useDeviceOp.getLoc(), newBoxwithDevicePtr.getType());
    [[maybe_unused]] fir::StoreOp newStoreOp = fir::StoreOp::create(
        rewriter, useDeviceOp.getLoc(), newBoxwithDevicePtr, newMemLoc);

    LLVM_DEBUG(llvm::dbgs()
               << "host_data region updated with new host descriptor "
                  "containing device pointer:\n"
               << "  box with device pointer: "
               << *newBoxwithDevicePtr.getDefiningOp() << "\n"
               << "  mem loc: " << *newMemLoc << "\n"
               << "  store op: " << *newStoreOp << "\n");

    // Replace all uses of the original `acc.use_device` operation inside the
    // `acc.host_data` region with the new memory location containing the box
    // with device pointer
    for (mlir::Operation *user : usersToUpdate)
      user->replaceUsesOfWith(useDeviceOp.getResult(), newMemLoc);

    assert(useDeviceOp.getResult().use_empty() &&
           "expected all uses of use_device to be replaced");
    rewriter.eraseOp(useDeviceOp);
    return true;
  }

  /// Canonicalize use_device operand that is a box type.
  /// Transforms:
  ///   %box = ... : !fir.box<!fir.array<?xi32>>
  ///   %dev_box = acc.use_device varPtr(%box : !fir.box<!fir.array<?xi32>>)
  ///   -> !fir.box<!fir.array<?xi32>>
  ///   acc.host_data dataOperands(%dev_box : !fir.box<!fir.array<?xi32>>) {
  ///     %addr = fir.box_addr %dev_box : (!fir.box<!fir.array<?xi32>>) ->
  ///     !fir.heap<!fir.array<?xi32>>
  ///     // use %addr
  ///   }
  /// into:
  ///   %box = ... : !fir.box<!fir.array<?xi32>>
  ///   %addr = fir.box_addr %box : (!fir.box<!fir.array<?xi32>>) ->
  ///   !fir.heap<!fir.array<?xi32>>
  ///   %dev_ptr = acc.use_device varPtr(%addr : !fir.heap<!fir.array<?xi32>>)
  ///   -> !fir.heap<!fir.array<?xi32>>
  ///   acc.host_data dataOperands(%dev_ptr : !fir.heap<!fir.array<?xi32>>) {
  ///     %new_box = fir.embox %dev_ptr ... : !fir.box<!fir.array<?xi32>>
  ///     %new_addr = fir.box_addr %new_box : (!fir.box<!fir.array<?xi32>>) ->
  ///     !fir.heap<!fir.array<?xi32>>
  ///     // use %new_addr instead of %addr
  ///   }
  bool hoistBox(PatternRewriter &rewriter, Value operand,
                acc::UseDeviceOp useDeviceOp,
                acc::HostDataOp hostDataOp) const {

    // Safety check: if the use_device operation is already using a box_addr
    // result, it means it has already been processed, so skip to avoid infinite
    // loop
    if (useDeviceOp.getVar().getDefiningOp<fir::BoxAddrOp>()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "ACCUseDeviceCanonicalizer: Skipping "
                    "already processed box use_device operation\n");
      return false;
    }

    // Collect users of the original `acc.use_device` operation that need to be
    // updated
    SmallVector<Operation *> usersToUpdate;
    collectUseDeviceUsersToUpdate(useDeviceOp, hostDataOp, usersToUpdate);

    // Get the ModuleOp before we erase useDeviceOp to avoid invalid reference
    ModuleOp mod = useDeviceOp->getParentOfType<ModuleOp>();

    rewriter.setInsertionPoint(useDeviceOp);
    // Extract the raw pointer from the box descriptor
    fir::BoxAddrOp boxAddr = fir::BoxAddrOp::create(
        rewriter, useDeviceOp.getLoc(), useDeviceOp.getVar());

    acc::UseDeviceOp newUseDeviceOp =
        createNewUseDeviceOp(rewriter, useDeviceOp, hostDataOp, boxAddr);

    // Set insertion point to the first op inside the host_data region
    rewriter.setInsertionPoint(&hostDataOp.getRegion().front().front());

    // Create a FirOpBuilder from the PatternRewriter using the module we got
    // earlier
    fir::FirOpBuilder builder(rewriter, mod);

    // Create a new host descriptor at the start of the host_data region
    // with the device pointer as the base address
    Value newBoxWithDevicePtr = fir::factory::getDescriptorWithNewBaseAddress(
        builder, useDeviceOp.getLoc(), useDeviceOp.getVar(),
        newUseDeviceOp.getResult());

    LLVM_DEBUG(llvm::dbgs()
               << "host_data region updated with new host descriptor "
                  "containing device pointer:\n"
               << "  box with device pointer: "
               << *newBoxWithDevicePtr.getDefiningOp() << "\n");

    // Replace all uses of the original `acc.use_device` operation inside the
    // `acc.host_data` region with the new box containing device pointer
    for (mlir::Operation *user : usersToUpdate)
      user->replaceUsesOfWith(useDeviceOp.getResult(), newBoxWithDevicePtr);

    assert(useDeviceOp.getResult().use_empty() &&
           "expected all uses of use_device to be replaced");
    rewriter.eraseOp(useDeviceOp);
    return true;
  }
};

class ACCUseDeviceCanonicalizer
    : public fir::acc::impl::ACCUseDeviceCanonicalizerBase<
          ACCUseDeviceCanonicalizer> {
public:
  void runOnOperation() override {
    MLIRContext *context = getOperation()->getContext();

    RewritePatternSet patterns(context);

    // Add the custom use_device canonicalization patterns
    patterns.insert<UseDeviceHostDataHoisting>(context);

    // Apply patterns greedily
    GreedyRewriteConfig config;
    // Prevent the pattern driver from merging blocks.
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    config.setUseTopDownTraversal(true);

    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> fir::acc::createACCUseDeviceCanonicalizerPass() {
  return std::make_unique<ACCUseDeviceCanonicalizer>();
}
