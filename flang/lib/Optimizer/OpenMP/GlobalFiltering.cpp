//===- GlobalFiltering.cpp ------------------------------------------------===//
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

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/OpenMP/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

namespace flangomp {
#define GEN_PASS_DEF_GLOBALFILTERINGPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

using namespace mlir;

namespace {
// TODO Remove this pass when AOMP moves to `clang-linker-wrapper` (instead of
// `clang-offload-packager`).
class GlobalFilteringPass
    : public flangomp::impl::GlobalFilteringPassBase<GlobalFilteringPass> {
public:
  GlobalFilteringPass() = default;

  void runOnOperation() override {
    auto op = dyn_cast<omp::OffloadModuleInterface>(getOperation());
    if (!op || !op.getIsTargetDevice())
      return;

    op->walk<WalkOrder::PreOrder>([&](fir::GlobalOp globalOp) {
      bool symbolUnused = true;
      SymbolTable::UseRange globalUses = *globalOp.getSymbolUses(op);
      for (SymbolTable::SymbolUse use : globalUses) {
        if (use.getUser() == globalOp)
          continue;
        symbolUnused = false;
        break;
      }

      // Look for declare target information in case this global is intended to
      // always exist on the device.
      auto declareTargetIface =
          llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(
              globalOp.getOperation());
      bool hostOnlySymbol = !declareTargetIface ||
                            !declareTargetIface.isDeclareTarget() ||
                            declareTargetIface.getDeclareTargetDeviceType() ==
                                omp::DeclareTargetDeviceType::host;

      // Remove unused host symbols with external linkage.
      if (symbolUnused && !globalOp.getLinkName() && hostOnlySymbol)
        globalOp.erase();
    });
  }
};
} // namespace
