//===- TemporaryStack.cpp ---- temporary stack runtime API calls ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/TemporaryStack.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/temporary-stack.h"

using namespace Fortran::runtime;

mlir::Value fir::runtime::genCreateValueStack(mlir::Location loc,
                                              fir::FirOpBuilder &builder) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(CreateValueStack)>(loc, builder);
  mlir::FunctionType funcType = func.getFunctionType();
  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, funcType.getInput(1));
  auto args = fir::runtime::createArguments(builder, loc, funcType, sourceFile,
                                            sourceLine);
  return fir::CallOp::create(builder, loc, func, args).getResult(0);
}

void fir::runtime::genPushValue(mlir::Location loc, fir::FirOpBuilder &builder,
                                mlir::Value opaquePtr, mlir::Value boxValue) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(PushValue)>(loc, builder);
  mlir::FunctionType funcType = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, funcType, opaquePtr,
                                            boxValue);
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genValueAt(mlir::Location loc, fir::FirOpBuilder &builder,
                              mlir::Value opaquePtr, mlir::Value i,
                              mlir::Value retValueBox) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(ValueAt)>(loc, builder);
  mlir::FunctionType funcType = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, funcType, opaquePtr,
                                            i, retValueBox);
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genDestroyValueStack(mlir::Location loc,
                                        fir::FirOpBuilder &builder,
                                        mlir::Value opaquePtr) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(DestroyValueStack)>(loc, builder);
  mlir::FunctionType funcType = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, funcType, opaquePtr);
  fir::CallOp::create(builder, loc, func, args);
}

mlir::Value fir::runtime::genCreateDescriptorStack(mlir::Location loc,
                                                   fir::FirOpBuilder &builder) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(CreateDescriptorStack)>(loc,
                                                                   builder);
  mlir::FunctionType funcType = func.getFunctionType();
  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, funcType.getInput(1));
  auto args = fir::runtime::createArguments(builder, loc, funcType, sourceFile,
                                            sourceLine);
  return fir::CallOp::create(builder, loc, func, args).getResult(0);
}

void fir::runtime::genPushDescriptor(mlir::Location loc,
                                     fir::FirOpBuilder &builder,
                                     mlir::Value opaquePtr,
                                     mlir::Value boxDescriptor) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(PushDescriptor)>(loc, builder);
  mlir::FunctionType funcType = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, funcType, opaquePtr,
                                            boxDescriptor);
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genDescriptorAt(mlir::Location loc,
                                   fir::FirOpBuilder &builder,
                                   mlir::Value opaquePtr, mlir::Value i,
                                   mlir::Value retDescriptorBox) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(DescriptorAt)>(loc, builder);
  mlir::FunctionType funcType = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, funcType, opaquePtr,
                                            i, retDescriptorBox);
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genDestroyDescriptorStack(mlir::Location loc,
                                             fir::FirOpBuilder &builder,
                                             mlir::Value opaquePtr) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(DestroyDescriptorStack)>(loc,
                                                                    builder);
  mlir::FunctionType funcType = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, funcType, opaquePtr);
  fir::CallOp::create(builder, loc, func, args);
}
