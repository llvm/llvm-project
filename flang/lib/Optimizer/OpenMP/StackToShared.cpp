//===- StackToShared.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements transforms to swap stack allocations on the target
// device with device shared memory where applicable.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"

namespace flangomp {
#define GEN_PASS_DEF_STACKTOSHAREDPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

using namespace mlir;

namespace {
class StackToSharedPass
    : public flangomp::impl::StackToSharedPassBase<StackToSharedPass> {
public:
  StackToSharedPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    OpBuilder builder(context);

    func::FuncOp funcOp = getOperation();
    auto offloadIface = funcOp->getParentOfType<omp::OffloadModuleInterface>();
    if (!offloadIface || !offloadIface.getIsTargetDevice())
      return;

    funcOp->walk([&](fir::AllocaOp allocaOp) {
      if (!shouldReplaceAlloca(*allocaOp))
        return;

      // Replace fir.alloca with omp.alloc_shared_mem.
      builder.setInsertionPoint(allocaOp);
      auto sharedAllocOp = omp::AllocSharedMemOp::create(
          builder, allocaOp->getLoc(), allocaOp.getResult().getType(),
          allocaOp.getInType(), allocaOp.getUniqNameAttr(),
          allocaOp.getBindcNameAttr(), allocaOp.getTypeparams(),
          allocaOp.getShape());
      allocaOp.replaceAllUsesWith(sharedAllocOp.getOperation());
      allocaOp.erase();

      // Create a new omp.free_shared_mem for the allocated buffer prior to
      // exiting the region.
      Block *allocaBlock = sharedAllocOp->getBlock();
      DominanceInfo domInfo;
      for (Block &block : sharedAllocOp->getParentRegion()->getBlocks()) {
        Operation *terminator = block.getTerminator();
        if (!terminator->hasSuccessors() &&
            domInfo.dominates(allocaBlock, &block)) {
          builder.setInsertionPoint(terminator);
          omp::FreeSharedMemOp::create(builder, sharedAllocOp.getLoc(),
                                       sharedAllocOp);
        }
      }
    });
  }

private:
  // TODO: Refactor the logic in `shouldReplaceAlloca` and `checkAllocaUses` to
  // be reusable by the MLIR to LLVM IR translation stage, as something very
  // similar is also implemented there to choose between allocas and device
  // shared memory allocations when processing OpenMP reductions, mapping and
  // privatization.

  // Decide whether to replace a fir.alloca with a pair of device shared memory
  // allocation/deallocation pair based on the location of the allocation and
  // its uses.
  //
  // In summary, it should be done whenever the allocation is placed outside any
  // parallel regions and inside either a target device function or a generic
  // kernel, while being used inside of a parallel region.
  bool shouldReplaceAlloca(Operation &op) {
    auto targetOp = op.getParentOfType<omp::TargetOp>();

    // It must be inside of a generic omp.target or in a target device function,
    // and not inside of omp.parallel.
    if (auto parallelOp = op.getParentOfType<omp::ParallelOp>()) {
      if (!targetOp || !targetOp->isProperAncestor(parallelOp))
        return false;
    }

    if (targetOp) {
      if (targetOp.getKernelExecFlags(targetOp.getInnermostCapturedOmpOp()) !=
          mlir::omp::TargetExecMode::generic)
        return false;
    } else {
      auto declTargetIface = dyn_cast<mlir::omp::DeclareTargetInterface>(
          *op.getParentOfType<func::FuncOp>());
      if (!declTargetIface || !declTargetIface.isDeclareTarget() ||
          declTargetIface.getDeclareTargetDeviceType() ==
              mlir::omp::DeclareTargetDeviceType::host)
        return false;
    }

    return checkAllocaUses(op.getUses());
  }

  // When a use takes place inside an omp.parallel region and it's not as a
  // private clause argument, or when it is a reduction argument passed to
  // omp.parallel, then the defining allocation is eligible for replacement with
  // shared memory.
  //
  // Only one of the uses needs to meet these conditions to return true.
  bool checkAllocaUses(const Operation::use_range &uses) {
    auto checkUse = [&](const OpOperand &use) {
      Operation *owner = use.getOwner();
      auto moduleOp = owner->getParentOfType<ModuleOp>();
      if (auto parallelOp = dyn_cast<omp::ParallelOp>(owner)) {
        if (llvm::is_contained(parallelOp.getReductionVars(), use.get()))
          return true;
      } else if (owner->getParentOfType<omp::ParallelOp>()) {
        // If it is used directly inside of a parallel region, it has to be
        // replaced unless the use is a private clause.
        if (auto argIface = dyn_cast<omp::BlockArgOpenMPOpInterface>(owner)) {
          if (auto privateSyms = llvm::cast_or_null<ArrayAttr>(
                  owner->getAttr("private_syms"))) {
            for (auto [var, sym] :
                 llvm::zip_equal(argIface.getPrivateVars(), privateSyms)) {
              if (var != use.get())
                continue;

              auto privateOp = cast<omp::PrivateClauseOp>(
                  moduleOp.lookupSymbol(cast<SymbolRefAttr>(sym)));
              return privateOp.getDataSharingType() !=
                     omp::DataSharingClauseType::Private;
            }
          }
        }
        return true;
      }
      return false;
    };

    // Check direct uses and also follow hlfir.declare uses.
    for (const OpOperand &use : uses) {
      if (auto declareOp = dyn_cast<hlfir::DeclareOp>(use.getOwner())) {
        if (checkAllocaUses(declareOp->getUses()))
          return true;
      } else if (checkUse(use)) {
        return true;
      }
    }

    return false;
  }
};
} // namespace
