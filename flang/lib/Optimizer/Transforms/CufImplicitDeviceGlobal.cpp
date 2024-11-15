//===-- CufOpConversion.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Common/Fortran.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Runtime/CUDA/common.h"
#include "flang/Runtime/allocatable.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace fir {
#define GEN_PASS_DEF_CUFIMPLICITDEVICEGLOBAL
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {

static void prepareImplicitDeviceGlobals(mlir::func::FuncOp funcOp,
                                         mlir::SymbolTable &symbolTable,
                                         bool onlyConstant = true) {
  auto cudaProcAttr{
      funcOp->getAttrOfType<cuf::ProcAttributeAttr>(cuf::getProcAttrName())};
  if (!cudaProcAttr || cudaProcAttr.getValue() == cuf::ProcAttribute::Host)
    return;
  for (auto addrOfOp : funcOp.getBody().getOps<fir::AddrOfOp>()) {
    if (auto globalOp = symbolTable.lookup<fir::GlobalOp>(
            addrOfOp.getSymbol().getRootReference().getValue())) {
      bool isCandidate{(onlyConstant ? globalOp.getConstant() : true) &&
                       !globalOp.getDataAttr()};
      if (isCandidate)
        globalOp.setDataAttrAttr(cuf::DataAttributeAttr::get(
            funcOp.getContext(), globalOp.getConstant()
                                     ? cuf::DataAttribute::Constant
                                     : cuf::DataAttribute::Device));
    }
  }
}

class CufImplicitDeviceGlobal
    : public fir::impl::CufImplicitDeviceGlobalBase<CufImplicitDeviceGlobal> {
public:
  void runOnOperation() override {
    mlir::Operation *op = getOperation();
    mlir::ModuleOp mod = mlir::dyn_cast<mlir::ModuleOp>(op);
    if (!mod)
      return signalPassFailure();

    mlir::SymbolTable symTable(mod);
    mod.walk([&](mlir::func::FuncOp funcOp) {
      prepareImplicitDeviceGlobals(funcOp, symTable);
      return mlir::WalkResult::advance();
    });
  }
};
} // namespace
