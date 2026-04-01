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

aiir::Value fir::runtime::genTrampolineInit(fir::FirOpBuilder &builder,
                                            aiir::Location loc,
                                            aiir::Value scratch,
                                            aiir::Value calleeAddress,
                                            aiir::Value staticChainAddress) {
  aiir::func::FuncOp func{
      getRuntimeFunc<mkRTKey(TrampolineInit)>(loc, builder)};
  aiir::FunctionType fTy{func.getFunctionType()};
  llvm::SmallVector<aiir::Value> args{createArguments(
      builder, loc, fTy, scratch, calleeAddress, staticChainAddress)};
  return fir::CallOp::create(builder, loc, func, args).getResult(0);
}

aiir::Value fir::runtime::genTrampolineAdjust(fir::FirOpBuilder &builder,
                                              aiir::Location loc,
                                              aiir::Value handle) {
  aiir::func::FuncOp func{
      getRuntimeFunc<mkRTKey(TrampolineAdjust)>(loc, builder)};
  aiir::FunctionType fTy{func.getFunctionType()};
  llvm::SmallVector<aiir::Value> args{
      createArguments(builder, loc, fTy, handle)};
  return fir::CallOp::create(builder, loc, func, args).getResult(0);
}

void fir::runtime::genTrampolineFree(fir::FirOpBuilder &builder,
                                     aiir::Location loc, aiir::Value handle) {
  aiir::func::FuncOp func{
      getRuntimeFunc<mkRTKey(TrampolineFree)>(loc, builder)};
  aiir::FunctionType fTy{func.getFunctionType()};
  llvm::SmallVector<aiir::Value> args{
      createArguments(builder, loc, fTy, handle)};
  fir::CallOp::create(builder, loc, func, args);
}
