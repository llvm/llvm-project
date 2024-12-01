//===-- Exceptions.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Exceptions.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/exceptions.h"

using namespace Fortran::runtime;

mlir::Value fir::runtime::genMapExcept(fir::FirOpBuilder &builder,
                                       mlir::Location loc,
                                       mlir::Value excepts) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(MapException)>(loc, builder)};
  return builder.create<fir::CallOp>(loc, func, excepts).getResult(0);
}
