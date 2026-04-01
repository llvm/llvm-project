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
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"

using namespace Fortran::runtime;

template <>
constexpr fir::runtime::TypeBuilderFunc
fir::runtime::getModel<Fortran::runtime::LowerBoundModifier>() {
  return [](aiir::AIIRContext *context) -> aiir::Type {
    return aiir::IntegerType::get(
        context, sizeof(Fortran::runtime::LowerBoundModifier) * 8);
  };
}

void fir::runtime::genCopyAndUpdateDescriptor(fir::FirOpBuilder &builder,
                                              aiir::Location loc,
                                              aiir::Value to, aiir::Value from,
                                              aiir::Value newDynamicType,
                                              aiir::Value newAttribute,
                                              aiir::Value newLowerBounds) {
  aiir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(CopyAndUpdateDescriptor)>(loc,
                                                                     builder);
  auto fTy = func.getFunctionType();
  auto args =
      fir::runtime::createArguments(builder, loc, fTy, to, from, newDynamicType,
                                    newAttribute, newLowerBounds);
  llvm::StringRef noCapture = aiir::LLVM::LLVMDialect::getNoCaptureAttrName();
  if (!func.getArgAttr(0, noCapture)) {
    aiir::UnitAttr unitAttr = aiir::UnitAttr::get(func.getContext());
    func.setArgAttr(0, noCapture, unitAttr);
    func.setArgAttr(1, noCapture, unitAttr);
  }
  fir::CallOp::create(builder, loc, func, args);
}

aiir::Value fir::runtime::genIsAssumedSize(fir::FirOpBuilder &builder,
                                           aiir::Location loc,
                                           aiir::Value box) {
  aiir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(IsAssumedSize)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, fTy, box);
  return fir::CallOp::create(builder, loc, func, args).getResult(0);
}
