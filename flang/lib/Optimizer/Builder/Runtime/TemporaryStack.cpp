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

aiir::Value fir::runtime::genCreateValueStack(aiir::Location loc,
                                              fir::FirOpBuilder &builder) {
  aiir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(CreateValueStack)>(loc, builder);
  aiir::FunctionType funcType = func.getFunctionType();
  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, funcType.getInput(1));
  auto args = fir::runtime::createArguments(builder, loc, funcType, sourceFile,
                                            sourceLine);
  return fir::CallOp::create(builder, loc, func, args).getResult(0);
}

void fir::runtime::genPushValue(aiir::Location loc, fir::FirOpBuilder &builder,
                                aiir::Value opaquePtr, aiir::Value boxValue) {
  aiir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(PushValue)>(loc, builder);
  aiir::FunctionType funcType = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, funcType, opaquePtr,
                                            boxValue);
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genValueAt(aiir::Location loc, fir::FirOpBuilder &builder,
                              aiir::Value opaquePtr, aiir::Value i,
                              aiir::Value retValueBox) {
  aiir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(ValueAt)>(loc, builder);
  aiir::FunctionType funcType = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, funcType, opaquePtr,
                                            i, retValueBox);
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genDestroyValueStack(aiir::Location loc,
                                        fir::FirOpBuilder &builder,
                                        aiir::Value opaquePtr) {
  aiir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(DestroyValueStack)>(loc, builder);
  aiir::FunctionType funcType = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, funcType, opaquePtr);
  fir::CallOp::create(builder, loc, func, args);
}

aiir::Value fir::runtime::genCreateDescriptorStack(aiir::Location loc,
                                                   fir::FirOpBuilder &builder) {
  aiir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(CreateDescriptorStack)>(loc,
                                                                   builder);
  aiir::FunctionType funcType = func.getFunctionType();
  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, funcType.getInput(1));
  auto args = fir::runtime::createArguments(builder, loc, funcType, sourceFile,
                                            sourceLine);
  return fir::CallOp::create(builder, loc, func, args).getResult(0);
}

void fir::runtime::genPushDescriptor(aiir::Location loc,
                                     fir::FirOpBuilder &builder,
                                     aiir::Value opaquePtr,
                                     aiir::Value boxDescriptor) {
  aiir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(PushDescriptor)>(loc, builder);
  aiir::FunctionType funcType = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, funcType, opaquePtr,
                                            boxDescriptor);
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genDescriptorAt(aiir::Location loc,
                                   fir::FirOpBuilder &builder,
                                   aiir::Value opaquePtr, aiir::Value i,
                                   aiir::Value retDescriptorBox) {
  aiir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(DescriptorAt)>(loc, builder);
  aiir::FunctionType funcType = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, funcType, opaquePtr,
                                            i, retDescriptorBox);
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genDestroyDescriptorStack(aiir::Location loc,
                                             fir::FirOpBuilder &builder,
                                             aiir::Value opaquePtr) {
  aiir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(DestroyDescriptorStack)>(loc,
                                                                    builder);
  aiir::FunctionType funcType = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, funcType, opaquePtr);
  fir::CallOp::create(builder, loc, func, args);
}
