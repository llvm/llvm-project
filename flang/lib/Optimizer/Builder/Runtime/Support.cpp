//===-- Support.cpp - generate support runtime API calls --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Support.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/support.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace Fortran::runtime;

template <>
constexpr fir::runtime::TypeBuilderFunc
fir::runtime::getModel<Fortran::runtime::LowerBoundModifier>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(
        context, sizeof(Fortran::runtime::LowerBoundModifier) * 8);
  };
}

void fir::runtime::genCopyAndUpdateDescriptor(fir::FirOpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value to, mlir::Value from,
                                              mlir::Value newDynamicType,
                                              mlir::Value newAttribute,
                                              mlir::Value newLowerBounds) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(CopyAndUpdateDescriptor)>(loc,
                                                                     builder);
  auto fTy = func.getFunctionType();
  auto args =
      fir::runtime::createArguments(builder, loc, fTy, to, from, newDynamicType,
                                    newAttribute, newLowerBounds);
  llvm::StringRef noCapture = mlir::LLVM::LLVMDialect::getNoCaptureAttrName();
  if (!func.getArgAttr(0, noCapture)) {
    mlir::UnitAttr unitAttr = mlir::UnitAttr::get(func.getContext());
    func.setArgAttr(0, noCapture, unitAttr);
    func.setArgAttr(1, noCapture, unitAttr);
  }
  builder.create<fir::CallOp>(loc, func, args);
}

mlir::Value fir::runtime::genIsAssumedSize(fir::FirOpBuilder &builder,
                                           mlir::Location loc,
                                           mlir::Value box) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(IsAssumedSize)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, fTy, box);
  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}
