//===-- FIRBuilder.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"

mlir::Value fir::FirOpBuilder::createIntegerConstant(mlir::Location loc,
                                                     mlir::Type ty,
                                                     std::int64_t cst) {
  return create<mlir::ConstantOp>(loc, ty, getIntegerAttr(ty, cst));
}

mlir::Value fir::FirOpBuilder::createConvert(mlir::Location loc,
                                             mlir::Type toTy, mlir::Value val) {
  if (val.getType() != toTy) {
    assert(!fir::isa_derived(toTy));
    return create<fir::ConvertOp>(loc, toTy, val);
  }
  return val;
}
