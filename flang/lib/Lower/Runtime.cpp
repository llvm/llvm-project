//===-- Runtime.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Runtime.h"
#include "../runtime/clock.h"
#include "../runtime/stop.h"
#include "RTBuilder.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/Support/BoxValue.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "llvm/ADT/SmallVector.h"

using namespace Fortran::runtime;
#define mkRTKey(X) mkKey(RTNAME(X))

static constexpr std::tuple<mkRTKey(DateAndTime), mkRTKey(FailImageStatement),
                            mkRTKey(PauseStatement),
                            mkRTKey(ProgramEndStatement),
                            mkRTKey(StopStatement), mkRTKey(StopStatementText)>
    newRTTable;

template <typename A>
static constexpr const char *getName() {
  return std::get<A>(newRTTable).name;
}

template <typename A>
static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
  return std::get<A>(newRTTable).getTypeModel();
}

template <typename RuntimeEntry>
static mlir::FuncOp genRuntimeFunction(mlir::Location loc,
                                       Fortran::lower::FirOpBuilder &builder) {
  auto name = getName<RuntimeEntry>();
  auto func = builder.getNamedFunction(name);
  if (func)
    return func;
  auto funTy = getTypeModel<RuntimeEntry>()(builder.getContext());
  func = builder.createFunction(loc, name, funTy);
  func.setAttr("fir.runtime", builder.getUnitAttr());
  return func;
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
  auto callee = genRuntimeFunction<mkRTKey(StopStatement)>(loc, builder);
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
  builder.create<fir::CallOp>(loc, callee, operands);
}

void Fortran::lower::genFailImageStatement(
    Fortran::lower::AbstractConverter &converter) {
  auto &bldr = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  auto callee = genRuntimeFunction<mkRTKey(FailImageStatement)>(loc, bldr);
  bldr.create<fir::CallOp>(loc, callee, llvm::None);
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
  auto &bldr = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  auto callee = genRuntimeFunction<mkRTKey(PauseStatement)>(loc, bldr);
  bldr.create<fir::CallOp>(loc, callee, llvm::None);
}

void Fortran::lower::genDateAndTime(Fortran::lower::FirOpBuilder &builder,
                                    mlir::Location loc,
                                    llvm::Optional<fir::CharBoxValue> date,
                                    llvm::Optional<fir::CharBoxValue> time,
                                    llvm::Optional<fir::CharBoxValue> zone) {
  auto callee = genRuntimeFunction<mkRTKey(DateAndTime)>(loc, builder);
  mlir::Type idxTy = builder.getIndexType();
  mlir::Value zero;
  auto splitArg = [&](llvm::Optional<fir::CharBoxValue> arg,
                      mlir::Value &buffer, mlir::Value &len) {
    if (arg) {
      buffer = arg->getBuffer();
      len = arg->getLen();
    } else {
      if (!zero)
        zero = builder.createIntegerConstant(loc, idxTy, 0);
      buffer = zero;
      len = zero;
    }
  };
  mlir::Value dateBuffer;
  mlir::Value dateLen;
  splitArg(date, dateBuffer, dateLen);
  mlir::Value timeBuffer;
  mlir::Value timeLen;
  splitArg(time, timeBuffer, timeLen);
  mlir::Value zoneBuffer;
  mlir::Value zoneLen;
  splitArg(zone, zoneBuffer, zoneLen);

  llvm::SmallVector<mlir::Value, 2> args{dateBuffer, timeBuffer, zoneBuffer,
                                         dateLen,    timeLen,    zoneLen};
  llvm::SmallVector<mlir::Value, 2> operands;
  for (const auto &op : llvm::zip(args, callee.getType().getInputs()))
    operands.emplace_back(
        builder.convertWithSemantics(loc, std::get<1>(op), std::get<0>(op)));
  builder.create<fir::CallOp>(loc, callee, operands);
}
