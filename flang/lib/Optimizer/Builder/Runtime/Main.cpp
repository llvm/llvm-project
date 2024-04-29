//===-- Main.cpp - generate main runtime API calls --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Main.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Runtime/main.h"
#include "flang/Runtime/stop.h"

using namespace Fortran::runtime;

/// Create a `int main(...)` that calls the Fortran entry point
void fir::runtime::genMain(fir::FirOpBuilder &builder, mlir::Location loc,
                           fir::GlobalOp &env) {
  auto *context = builder.getContext();
  auto argcTy = builder.getDefaultIntegerType();
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(context);

  // void ProgramStart(int argc, char** argv, char** envp,
  //                   _QQEnvironmentDefaults* env)
  auto startFn = builder.createFunction(
      loc, RTNAME_STRING(ProgramStart),
      mlir::FunctionType::get(context, {argcTy, ptrTy, ptrTy, ptrTy}, {}));
  // void ProgramStop()
  auto stopFn =
      builder.createFunction(loc, RTNAME_STRING(ProgramEndStatement),
                             mlir::FunctionType::get(context, {}, {}));

  // int main(int argc, char** argv, char** envp)
  auto mainFn = builder.createFunction(
      loc, "main",
      mlir::FunctionType::get(context, {argcTy, ptrTy, ptrTy}, argcTy));
  // void _QQmain()
  auto qqMainFn = builder.createFunction(
      loc, "_QQmain", mlir::FunctionType::get(context, {}, {}));

  mainFn.setPublic();

  auto *block = mainFn.addEntryBlock();
  mlir::OpBuilder::InsertionGuard insertGuard(builder);
  builder.setInsertionPointToStart(block);

  llvm::SmallVector<mlir::Value, 4> args(block->getArguments());
  auto envAddr =
      builder.create<fir::AddrOfOp>(loc, env.getType(), env.getSymbol());
  args.push_back(envAddr);

  builder.create<fir::CallOp>(loc, startFn, args);
  builder.create<fir::CallOp>(loc, qqMainFn);
  builder.create<fir::CallOp>(loc, stopFn);

  mlir::Value ret = builder.createIntegerConstant(loc, argcTy, 0);
  builder.create<mlir::func::ReturnOp>(loc, ret);
}
