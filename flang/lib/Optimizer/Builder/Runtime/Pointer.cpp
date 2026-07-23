//===-- Pointer.cpp -- generate pointer runtime API calls------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Pointer.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/pointer.h"

using namespace Fortran::runtime;

void fir::runtime::genPointerAssociateScalar(fir::FirOpBuilder &builder,
                                             mlir::Location loc,
                                             mlir::Value desc,
                                             mlir::Value target) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(PointerAssociateScalar)>(loc,
                                                                    builder)};
  mlir::FunctionType fTy{func.getFunctionType()};
  llvm::SmallVector<mlir::Value> args{
      fir::runtime::createArguments(builder, loc, fTy, desc, target)};
  fir::CallOp::create(builder, loc, func, args);
}
