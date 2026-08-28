//===- UnimplementedDeviceCheck.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Diagnose not-yet-implemented target device cases.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/OpenMP/Passes.h"

#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace flangomp {
#define GEN_PASS_DEF_UNIMPLEMENTEDDEVICECHECKPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

/// Check whether all uses of the given symbol inside of the module are only
/// intended for the host.
static bool allUsesInHostCode(Operation *moduleOp, SymbolOpInterface symOp) {
  if (auto symUses = symOp.getSymbolUses(moduleOp)) {
    for (auto symUse : symUses.value()) {
      Operation *symUser = symUse.getUser();
      if (!symUser)
        continue;

      if (symUser->getParentOfType<omp::TargetOp>())
        return false;

      if (auto declareTargetOp =
              symUser->getParentOfType<omp::DeclareTargetInterface>()) {
        if (declareTargetOp.isDeclareTarget() &&
            declareTargetOp.getDeclareTargetDeviceType() !=
                omp::DeclareTargetDeviceType::host)
          return false;
      }
    }
  }
  return true;
}

/// Emit not-yet-implemented errors for reductions over dynamically-shaped
/// arrays.
static LogicalResult checkReduction(omp::DeclareReductionOp reductionOp) {
  if (!reductionOp.getByrefElementType())
    return success();

  auto seqTy = dyn_cast<fir::SequenceType>(*reductionOp.getByrefElementType());

  bool isByRefReductionSupported =
      !seqTy || !fir::sequenceWithNonConstantShape(seqTy);

  if (!isByRefReductionSupported)
    return reductionOp.emitError()
           << "not yet implemented: Reduction of dynamically-shaped arrays on "
              "the GPU.";

  return success();
}

namespace {
class UnimplementedDeviceCheckPass
    : public flangomp::impl::UnimplementedDeviceCheckPassBase<
          UnimplementedDeviceCheckPass> {

  void runOnOperation() override {
    // Only run checks when compiling for a target device.
    auto op = dyn_cast<omp::OffloadModuleInterface>(*getOperation());
    if (!op || !op.getIsTargetDevice())
      return;

    bool errorsEmitted = false;

    if (op.getIsGPU()) {
      op->walk([&](omp::DeclareReductionOp reductionOp) {
        // Only check reductions that aren't exclusively used on the host.
        if (op.getIsGPU() && !allUsesInHostCode(op, reductionOp)) {
          if (failed(checkReduction(reductionOp)))
            errorsEmitted = true;
        }
      });
    }

    if (errorsEmitted)
      signalPassFailure();

    markAllAnalysesPreserved();
  }
};
} // namespace
