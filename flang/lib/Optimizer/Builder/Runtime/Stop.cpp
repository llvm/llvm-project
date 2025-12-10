//===-- Stop.h - generate stop runtime API calls ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Stop.h"
#include "flang/Lower/Runtime.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/stop.h"

using namespace Fortran::runtime;

/// Runtime calls that do not return to the caller indicate this condition by
/// terminating the current basic block with an unreachable op.
static void genUnreachable(fir::FirOpBuilder &builder, mlir::Location loc) {
  mlir::Block *curBlock = builder.getBlock();
#if 0
  mlir::Operation *parentOp = curBlock->getParentOp();
  if (parentOp->getDialect()->getNamespace() ==
      mlir::omp::OpenMPDialect::getDialectNamespace())
    Fortran::lower::genOpenMPTerminator(builder, parentOp, loc);
  else if (Fortran::lower::isInsideOpenACCComputeConstruct(builder))
    Fortran::lower::genOpenACCTerminator(builder, parentOp, loc);
  else
#endif
  fir::UnreachableOp::create(builder, loc);
  mlir::Block *newBlock = curBlock->splitBlock(builder.getInsertionPoint());
  builder.setInsertionPointToStart(newBlock);
}

void fir::runtime::genExit(fir::FirOpBuilder &builder, mlir::Location loc,
                           mlir::Value status) {
  auto exitFunc = fir::runtime::getRuntimeFunc<mkRTKey(Exit)>(loc, builder);
  exitFunc->setAttr("noreturn", builder.getUnitAttr());
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, exitFunc.getFunctionType(), status);
  fir::CallOp::create(builder, loc, exitFunc, args);
  genUnreachable(builder, loc);
}

void fir::runtime::genAbort(fir::FirOpBuilder &builder, mlir::Location loc) {
  mlir::func::FuncOp abortFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(Abort)>(loc, builder);
  fir::CallOp::create(builder, loc, abortFunc, mlir::ValueRange{});
  genUnreachable(builder, loc);
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
  fir::CallOp::create(builder, loc, crashFunc, args);
}
