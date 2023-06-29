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

using namespace fir;
using namespace mlir;

namespace {
class OMPFunctionFilteringPass
    : public fir::impl::OMPFunctionFilteringBase<OMPFunctionFilteringPass> {
public:
  OMPFunctionFilteringPass() = default;

  void runOnOperation() override {
    auto op = dyn_cast<omp::OffloadModuleInterface>(getOperation());
    if (!op)
      return;

    bool isDeviceCompilation = op.getIsTargetDevice();
    op->walk<WalkOrder::PostOrder>([&](func::FuncOp funcOp) {
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

      if ((isDeviceCompilation &&
           declareType == omp::DeclareTargetDeviceType::host) ||
          (!isDeviceCompilation &&
           declareType == omp::DeclareTargetDeviceType::nohost))
        funcOp->erase();
    });
  }
};
} // namespace

std::unique_ptr<Pass> fir::createOMPFunctionFilteringPass() {
  return std::make_unique<OMPFunctionFilteringPass>();
}
