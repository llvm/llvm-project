//===-- Main.cpp - generate main runtime API calls --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Main.h"
#include "flang/Lower/EnvironmentDefault.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/EnvironmentDefaults.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Runtime/main.h"
#include "flang/Runtime/stop.h"

using namespace Fortran::runtime;

/// Create a `int main(...)` that calls the Fortran entry point
void fir::runtime::genMain(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const std::vector<Fortran::lower::EnvironmentDefault> &defs) {
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

  // Create the list of any environment defaults for the runtime to set. The
  // runtime default list is only created if there is a main program to ensure
  // it only happens once and to provide consistent results if multiple files
  // are compiled separately.
  auto env = fir::runtime::genEnvironmentDefaults(builder, loc, defs);

  llvm::SmallVector<mlir::Value, 4> args(block->getArguments());
  args.push_back(env);

  builder.create<fir::CallOp>(loc, startFn, args);
  builder.create<fir::CallOp>(loc, qqMainFn);
  builder.create<fir::CallOp>(loc, stopFn);

  mlir::Value ret = builder.createIntegerConstant(loc, argcTy, 0);
  builder.create<mlir::func::ReturnOp>(loc, ret);
}
