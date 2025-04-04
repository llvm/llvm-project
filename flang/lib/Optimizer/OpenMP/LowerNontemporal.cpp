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
    if (!simdOp.getNontemporalVars().empty()) {
      llvm::SmallVector<mlir::Value> nontemporalOrigVars;
      mlir::OperandRange nontemporals = simdOp.getNontemporalVars();
      for (mlir::Value nontemporal : nontemporals) {
        nontemporalOrigVars.push_back(nontemporal);
      }
      std::function<mlir::Value(mlir::Value)> getBaseOperand =
          [&](mlir::Value operand) -> mlir::Value {
        if (mlir::isa<fir::DeclareOp>(operand.getDefiningOp()))
          return operand;
        else if (auto arrayCoorOp = llvm::dyn_cast<fir::ArrayCoorOp>(
                     operand.getDefiningOp())) {
          return getBaseOperand(arrayCoorOp.getMemref());
        } else if (auto boxAddrOp = llvm::dyn_cast<fir::BoxAddrOp>(
                       operand.getDefiningOp())) {
          return getBaseOperand(boxAddrOp.getVal());
        } else if (auto loadOp =
                       llvm::dyn_cast<fir::LoadOp>(operand.getDefiningOp())) {
          return getBaseOperand(loadOp.getMemref());
        } else {
          return operand;
        }
      };
      simdOp->walk([&](Operation *op) {
        mlir::Value Operand = nullptr;
        if (auto loadOp = llvm::dyn_cast<fir::LoadOp>(op)) {
          Operand = loadOp.getMemref();
        } else if (auto storeOp = llvm::dyn_cast<fir::StoreOp>(op)) {
          Operand = storeOp.getMemref();
        }
        if (Operand && !(fir::isAllocatableType(Operand.getType()) ||
                         fir::isPointerType((Operand.getType())))) {
          Operand = getBaseOperand(Operand);
          if (is_contained(nontemporalOrigVars, Operand)) {
            // Set the attribute
            op->setAttr("nontemporal", UnitAttr::get(op->getContext()));
          }
        }
      });
    }
  }
  void runOnOperation() override {
    Operation *op = getOperation();
    op->walk([&](omp::SimdOp simdOp) { addNonTemporalAttr(simdOp); });
  }
};
} // namespace