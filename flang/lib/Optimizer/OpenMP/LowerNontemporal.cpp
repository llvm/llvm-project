//===- LowerNontemporal.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Add nontemporal attributes to load and stores of variables marked as
// nontemporal.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

using namespace mlir;

namespace flangomp {
#define GEN_PASS_DEF_LOWERNONTEMPORALPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

namespace {
class LowerNontemporalPass
    : public flangomp::impl::LowerNontemporalPassBase<LowerNontemporalPass> {
  void addNonTemporalAttr(omp::SimdOp simdOp) {
    if (simdOp.getNontemporalVars().empty())
      return;

    std::function<mlir::Value(mlir::Value)> getBaseOperand =
        [&](mlir::Value operand) -> mlir::Value {
      if (mlir::isa<mlir::BlockArgument>(operand) ||
          (mlir::isa<fir::AllocaOp>(operand.getDefiningOp())) ||
          (mlir::isa<fir::DeclareOp>(operand.getDefiningOp())))
        return operand;

      Operation *definingOp = operand.getDefiningOp();
      if (definingOp) {
        for (Value srcOp : definingOp->getOperands()) {
          return getBaseOperand(srcOp);
        }
      }
      return operand;
    };

    // walk through the operations and mark the load and store as nontemporal
    simdOp->walk([&](Operation *op) {
      mlir::Value operand = nullptr;

      if (auto loadOp = llvm::dyn_cast<fir::LoadOp>(op))
        operand = loadOp.getMemref();
      else if (auto storeOp = llvm::dyn_cast<fir::StoreOp>(op))
        operand = storeOp.getMemref();

      if (operand && !(fir::isAllocatableType(operand.getType()) ||
                       fir::isPointerType((operand.getType())))) {
        operand = getBaseOperand(operand);

        if (llvm::is_contained(simdOp.getNontemporalVars(), operand)) {
          if (auto loadOp = llvm::dyn_cast<fir::LoadOp>(op))
            loadOp.setNontemporal(true);
          else if (auto storeOp = llvm::dyn_cast<fir::StoreOp>(op))
            storeOp.setNontemporal(true);
        }
      }
    });
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    op->walk([&](omp::SimdOp simdOp) { addNonTemporalAttr(simdOp); });
  }
};
} // namespace
