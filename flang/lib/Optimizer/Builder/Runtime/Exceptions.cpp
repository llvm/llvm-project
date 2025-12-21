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
  return fir::CallOp::create(builder, loc, func, excepts).getResult(0);
}

void fir::runtime::genFeclearexcept(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value excepts) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(feclearexcept)>(loc, builder)};
  fir::CallOp::create(builder, loc, func, excepts);
}

void fir::runtime::genFeraiseexcept(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value excepts) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(feraiseexcept)>(loc, builder)};
  fir::CallOp::create(builder, loc, func, excepts);
}

mlir::Value fir::runtime::genFetestexcept(fir::FirOpBuilder &builder,
                                          mlir::Location loc,
                                          mlir::Value excepts) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(fetestexcept)>(loc, builder)};
  return fir::CallOp::create(builder, loc, func, excepts).getResult(0);
}

void fir::runtime::genFedisableexcept(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Value excepts) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(fedisableexcept)>(loc, builder)};
  fir::CallOp::create(builder, loc, func, excepts);
}

void fir::runtime::genFeenableexcept(fir::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Value excepts) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(feenableexcept)>(loc, builder)};
  fir::CallOp::create(builder, loc, func, excepts);
}

mlir::Value fir::runtime::genFegetexcept(fir::FirOpBuilder &builder,
                                         mlir::Location loc) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(fegetexcept)>(loc, builder)};
  return fir::CallOp::create(builder, loc, func).getResult(0);
}

mlir::Value fir::runtime::genSupportHalting(fir::FirOpBuilder &builder,
                                            mlir::Location loc,
                                            mlir::Value excepts) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(SupportHalting)>(loc, builder)};
  return fir::CallOp::create(builder, loc, func, excepts).getResult(0);
}

mlir::Value fir::runtime::genGetUnderflowMode(fir::FirOpBuilder &builder,
                                              mlir::Location loc) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(GetUnderflowMode)>(loc, builder)};
  return fir::CallOp::create(builder, loc, func).getResult(0);
}

void fir::runtime::genSetUnderflowMode(fir::FirOpBuilder &builder,
                                       mlir::Location loc, mlir::Value flag) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(SetUnderflowMode)>(loc, builder)};
  fir::CallOp::create(builder, loc, func, flag);
}

mlir::Value fir::runtime::genGetModesTypeSize(fir::FirOpBuilder &builder,
                                              mlir::Location loc) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(GetModesTypeSize)>(loc, builder)};
  return fir::CallOp::create(builder, loc, func).getResult(0);
}

mlir::Value fir::runtime::genGetStatusTypeSize(fir::FirOpBuilder &builder,
                                               mlir::Location loc) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(GetStatusTypeSize)>(loc, builder)};
  return fir::CallOp::create(builder, loc, func).getResult(0);
}
