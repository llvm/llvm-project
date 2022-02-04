//===-- Runtime.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Runtime.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Runtime/stop.h"
#include "flang/Semantics/tools.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-lower-runtime"

using namespace Fortran::runtime;

/// Runtime calls that do not return to the caller indicate this condition by
/// terminating the current basic block with an unreachable op.
static void genUnreachable(fir::FirOpBuilder &builder, mlir::Location loc) {
  builder.create<fir::UnreachableOp>(loc);
  mlir::Block *newBlock =
      builder.getBlock()->splitBlock(builder.getInsertionPoint());
  builder.setInsertionPointToStart(newBlock);
}

//===----------------------------------------------------------------------===//
// Misc. Fortran statements that lower to runtime calls
//===----------------------------------------------------------------------===//

void Fortran::lower::genStopStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::StopStmt &stmt) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  llvm::SmallVector<mlir::Value> operands;
  mlir::FuncOp callee;
  mlir::FunctionType calleeType;
  // First operand is stop code (zero if absent)
  if (const auto &code =
          std::get<std::optional<Fortran::parser::StopCode>>(stmt.t)) {
    auto expr = converter.genExprValue(*Fortran::semantics::GetExpr(*code));
    LLVM_DEBUG(llvm::dbgs() << "stop expression: "; expr.dump();
               llvm::dbgs() << '\n');
    expr.match(
        [&](const fir::CharBoxValue &x) {
          TODO(loc, "STOP CharBoxValue first operand not lowered yet");
        },
        [&](fir::UnboxedValue x) {
          callee = fir::runtime::getRuntimeFunc<mkRTKey(StopStatement)>(
              loc, builder);
          calleeType = callee.getType();
          mlir::Value cast =
              builder.createConvert(loc, calleeType.getInput(0), x);
          operands.push_back(cast);
        },
        [&](auto) {
          mlir::emitError(loc, "unhandled expression in STOP");
          std::exit(1);
        });
  } else {
    callee = fir::runtime::getRuntimeFunc<mkRTKey(StopStatement)>(loc, builder);
    calleeType = callee.getType();
    operands.push_back(
        builder.createIntegerConstant(loc, calleeType.getInput(0), 0));
  }

  // Second operand indicates ERROR STOP
  bool isError = std::get<Fortran::parser::StopStmt::Kind>(stmt.t) ==
                 Fortran::parser::StopStmt::Kind::ErrorStop;
  operands.push_back(builder.createIntegerConstant(
      loc, calleeType.getInput(operands.size()), isError));

  // Third operand indicates QUIET (default to false).
  if (const auto &quiet =
          std::get<std::optional<Fortran::parser::ScalarLogicalExpr>>(stmt.t)) {
    const SomeExpr *expr = Fortran::semantics::GetExpr(*quiet);
    assert(expr && "failed getting typed expression");
    mlir::Value q = fir::getBase(converter.genExprValue(*expr));
    operands.push_back(
        builder.createConvert(loc, calleeType.getInput(operands.size()), q));
  } else {
    operands.push_back(builder.createIntegerConstant(
        loc, calleeType.getInput(operands.size()), 0));
  }

  builder.create<fir::CallOp>(loc, callee, operands);
  genUnreachable(builder, loc);
}

void Fortran::lower::genPauseStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::PauseStmt &) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  mlir::FuncOp callee =
      fir::runtime::getRuntimeFunc<mkRTKey(PauseStatement)>(loc, builder);
  builder.create<fir::CallOp>(loc, callee, llvm::None);
}
