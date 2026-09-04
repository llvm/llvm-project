//===-- Trampoline.cpp - Runtime trampoline pool builder --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Trampoline.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/trampoline.h"

using namespace Fortran::runtime;
using namespace fir::runtime;

mlir::Value fir::runtime::genTrampolineInit(fir::FirOpBuilder &builder,
                                            mlir::Location loc,
                                            mlir::Value scratch,
                                            mlir::Value calleeAddress,
                                            mlir::Value staticChainAddress) {
  mlir::func::FuncOp func{
      getRuntimeFunc<mkRTKey(TrampolineInit)>(loc, builder)};
  mlir::FunctionType fTy{func.getFunctionType()};
  llvm::SmallVector<mlir::Value> args{createArguments(
      builder, loc, fTy, scratch, calleeAddress, staticChainAddress)};
  return fir::CallOp::create(builder, loc, func, args).getResult(0);
}

mlir::Value fir::runtime::genTrampolineAdjust(fir::FirOpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value handle) {
  mlir::func::FuncOp func{
      getRuntimeFunc<mkRTKey(TrampolineAdjust)>(loc, builder)};
  mlir::FunctionType fTy{func.getFunctionType()};
  llvm::SmallVector<mlir::Value> args{
      createArguments(builder, loc, fTy, handle)};
  return fir::CallOp::create(builder, loc, func, args).getResult(0);
}

void fir::runtime::genTrampolineFree(fir::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Value handle) {
  mlir::func::FuncOp func{
      getRuntimeFunc<mkRTKey(TrampolineFree)>(loc, builder)};
  mlir::FunctionType fTy{func.getFunctionType()};
  llvm::SmallVector<mlir::Value> args{
      createArguments(builder, loc, fTy, handle)};
  fir::CallOp::create(builder, loc, func, args);
}
