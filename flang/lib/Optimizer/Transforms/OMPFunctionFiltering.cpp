//===- OMPFunctionFiltering.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements transforms to filter out functions intended for the host
// when compiling for the device and vice versa.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

namespace fir {
#define GEN_PASS_DEF_OMPFUNCTIONFILTERING
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace mlir;

namespace {
class OMPFunctionFilteringPass
    : public fir::impl::OMPFunctionFilteringBase<OMPFunctionFilteringPass> {
public:
  OMPFunctionFilteringPass() = default;

  void runOnOperation() override {
    auto op = dyn_cast<omp::OffloadModuleInterface>(getOperation());
    if (!op || !op.getIsTargetDevice())
      return;

    op->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
      // Do not filter functions with target regions inside, because they have
      // to be available for both host and device so that regular and reverse
      // offloading can be supported.
      bool hasTargetRegion =
          funcOp
              ->walk<WalkOrder::PreOrder>(
                  [&](omp::TargetOp) { return WalkResult::interrupt(); })
              .wasInterrupted();
      if (hasTargetRegion)
        return;

      omp::DeclareTargetDeviceType declareType =
          omp::DeclareTargetDeviceType::host;
      auto declareTargetOp =
          dyn_cast<omp::DeclareTargetInterface>(funcOp.getOperation());
      if (declareTargetOp && declareTargetOp.isDeclareTarget())
        declareType = declareTargetOp.getDeclareTargetDeviceType();

      // Filtering a function here means removing its body and explicitly
      // setting its omp.declare_target attribute, so that following
      // translation/lowering/transformation passes will skip processing its
      // contents, but preventing the calls to undefined symbols that could
      // result if the function were deleted. The second stage of function
      // filtering, at the MLIR to LLVM IR translation level, will remove these
      // from the IR thanks to the mismatch between the omp.declare_target
      // attribute and the target device.
      if (declareType == omp::DeclareTargetDeviceType::host) {
        funcOp.eraseBody();
        funcOp.setVisibility(SymbolTable::Visibility::Private);
        if (declareTargetOp)
          declareTargetOp.setDeclareTarget(declareType,
                                           omp::DeclareTargetCaptureClause::to);
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> fir::createOMPFunctionFilteringPass() {
  return std::make_unique<OMPFunctionFilteringPass>();
}
