//===-- Runtime.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Runtime.h"
#include "../runtime/stop.h"
#include "RTBuilder.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "llvm/ADT/SmallVector.h"

#define MakeRuntimeEntry(X) mkKey(RTNAME(X))

template <typename RuntimeEntry>
static mlir::FuncOp genRuntimeFunction(mlir::Location loc,
                                       Fortran::lower::FirOpBuilder &builder) {
  auto func = builder.getNamedFunction(RuntimeEntry::name);
  if (func)
    return func;
  auto funTy = RuntimeEntry::getTypeModel()(builder.getContext());
  func = builder.createFunction(loc, RuntimeEntry::name, funTy);
  func.setAttr("fir.runtime", builder.getUnitAttr());
  return func;
}

static mlir::FuncOp
genStopStatementRuntime(mlir::Location loc,
                        Fortran::lower::FirOpBuilder &builder) {
  return genRuntimeFunction<MakeRuntimeEntry(StopStatement)>(loc, builder);
}

static mlir::FuncOp
genStopStatementTextRuntime(mlir::Location loc,
                            Fortran::lower::FirOpBuilder &builder) {
  return genRuntimeFunction<MakeRuntimeEntry(StopStatementText)>(loc, builder);
}

static mlir::FuncOp
genProgramEndStatementRuntime(mlir::Location loc,
                              Fortran::lower::FirOpBuilder &builder) {
  return genRuntimeFunction<MakeRuntimeEntry(ProgramEndStatement)>(loc,
                                                                   builder);
}

// TODO: We don't have runtime library support for various features. When they
// are encountered, we emit an error message and exit immediately.
static void noRuntimeSupport(mlir::Location loc, llvm::StringRef stmt) {
  mlir::emitError(loc, "There is no runtime support for ")
      << stmt << " statement.\n";
  std::exit(1);
}

//===----------------------------------------------------------------------===//
// Misc. Fortran statements that lower to runtime calls
//===----------------------------------------------------------------------===//

void Fortran::lower::genStopStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::StopStmt &stmt) {
  auto &builder = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  auto callee = genStopStatementRuntime(loc, builder);
  auto calleeType = callee.getType();
  llvm::SmallVector<mlir::Value, 8> operands;
  assert(calleeType.getNumInputs() == 3 &&
         "expected 3 arguments in STOP runtime");
  // First operand is stop code (zero if absent)
  if (const auto &code =
          std::get<std::optional<Fortran::parser::StopCode>>(stmt.t)) {
    auto expr = Fortran::semantics::GetExpr(*code);
    assert(expr && "failed getting typed expression");
    operands.push_back(converter.genExprValue(*expr));
  } else {
    operands.push_back(
        builder.createIntegerConstant(loc, calleeType.getInput(0), 0));
  }
  // Second operand indicates ERROR STOP
  bool isError = std::get<Fortran::parser::StopStmt::Kind>(stmt.t) ==
                 Fortran::parser::StopStmt::Kind::ErrorStop;
  operands.push_back(
      builder.createIntegerConstant(loc, calleeType.getInput(1), isError));

  // Third operand indicates QUIET (default to false).
  if (const auto &quiet =
          std::get<std::optional<Fortran::parser::ScalarLogicalExpr>>(stmt.t)) {
    auto expr = Fortran::semantics::GetExpr(*quiet);
    assert(expr && "failed getting typed expression");
    operands.push_back(converter.genExprValue(*expr));
  } else {
    operands.push_back(
        builder.createIntegerConstant(loc, calleeType.getInput(2), 0));
  }

  // Cast operands in case they have different integer/logical types
  // compare to runtime.
  auto i = 0;
  for (auto &op : operands) {
    auto type = calleeType.getInput(i++);
    op = builder.createConvert(loc, type, op);
  }
  builder.create<mlir::CallOp>(loc, callee, operands);
}

void Fortran::lower::genFailImageStatement(
    Fortran::lower::AbstractConverter &converter) {
  auto &bldr = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  auto callee =
      genRuntimeFunction<MakeRuntimeEntry(FailImageStatement)>(loc, bldr);
  bldr.create<mlir::CallOp>(loc, callee, llvm::None);
}

void Fortran::lower::genEventPostStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::EventPostStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "EVENT POST");
}

void Fortran::lower::genEventWaitStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::EventWaitStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "EVENT WAIT");
}

void Fortran::lower::genLockStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::LockStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "LOCK");
}

void Fortran::lower::genUnlockStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::UnlockStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "UNLOCK");
}

void Fortran::lower::genSyncAllStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncAllStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "SYNC ALL");
}

void Fortran::lower::genSyncImagesStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncImagesStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "SYNC IMAGES");
}

void Fortran::lower::genSyncMemoryStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncMemoryStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "SYNC MEMORY");
}

void Fortran::lower::genSyncTeamStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncTeamStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "SYNC TEAM");
}

void Fortran::lower::genPauseStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::PauseStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "PAUSE");
}
