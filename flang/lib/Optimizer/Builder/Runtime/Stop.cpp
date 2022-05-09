//===-- Stop.h - generate stop runtime API calls ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Stop.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/stop.h"

using namespace Fortran::runtime;

void fir::runtime::genExit(fir::FirOpBuilder &builder, mlir::Location loc,
                           mlir::Value status) {
  auto exitFunc = fir::runtime::getRuntimeFunc<mkRTKey(Exit)>(loc, builder);
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, exitFunc.getFunctionType(), status);
  builder.create<fir::CallOp>(loc, exitFunc, args);
}

void fir::runtime::genReportFatalUserError(fir::FirOpBuilder &builder,
                                           mlir::Location loc,
                                           llvm::StringRef message) {
  mlir::func::FuncOp crashFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(ReportFatalUserError)>(loc, builder);
  mlir::FunctionType funcTy = crashFunc.getFunctionType();
  mlir::Value msgVal = fir::getBase(
      fir::factory::createStringLiteral(builder, loc, message.str() + '\0'));
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, funcTy.getInput(2));
  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, funcTy, msgVal, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, crashFunc, args);
}
