//===-- Runtime.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Runtime.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Runtime/misc-intrinsic.h"
#include "flang/Runtime/pointer.h"
#include "flang/Runtime/random.h"
#include "flang/Runtime/stop.h"
#include "flang/Runtime/time-intrinsic.h"
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
  Fortran::lower::StatementContext stmtCtx;
  llvm::SmallVector<mlir::Value> operands;
  mlir::func::FuncOp callee;
  mlir::FunctionType calleeType;
  // First operand is stop code (zero if absent)
  if (const auto &code =
          std::get<std::optional<Fortran::parser::StopCode>>(stmt.t)) {
    auto expr =
        converter.genExprValue(*Fortran::semantics::GetExpr(*code), stmtCtx);
    LLVM_DEBUG(llvm::dbgs() << "stop expression: "; expr.dump();
               llvm::dbgs() << '\n');
    expr.match(
        [&](const fir::CharBoxValue &x) {
          callee = fir::runtime::getRuntimeFunc<mkRTKey(StopStatementText)>(
              loc, builder);
          calleeType = callee.getFunctionType();
          // Creates a pair of operands for the CHARACTER and its LEN.
          operands.push_back(
              builder.createConvert(loc, calleeType.getInput(0), x.getAddr()));
          operands.push_back(
              builder.createConvert(loc, calleeType.getInput(1), x.getLen()));
        },
        [&](fir::UnboxedValue x) {
          callee = fir::runtime::getRuntimeFunc<mkRTKey(StopStatement)>(
              loc, builder);
          calleeType = callee.getFunctionType();
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
    calleeType = callee.getFunctionType();
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
    mlir::Value q = fir::getBase(converter.genExprValue(*expr, stmtCtx));
    operands.push_back(
        builder.createConvert(loc, calleeType.getInput(operands.size()), q));
  } else {
    operands.push_back(builder.createIntegerConstant(
        loc, calleeType.getInput(operands.size()), 0));
  }

  builder.create<fir::CallOp>(loc, callee, operands);
  genUnreachable(builder, loc);
}

void Fortran::lower::genFailImageStatement(
    Fortran::lower::AbstractConverter &converter) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  mlir::func::FuncOp callee =
      fir::runtime::getRuntimeFunc<mkRTKey(FailImageStatement)>(loc, builder);
  builder.create<fir::CallOp>(loc, callee, llvm::None);
  genUnreachable(builder, loc);
}

void Fortran::lower::genEventPostStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::EventPostStmt &) {
  TODO(converter.getCurrentLocation(), "EVENT POST runtime");
}

void Fortran::lower::genEventWaitStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::EventWaitStmt &) {
  TODO(converter.getCurrentLocation(), "EVENT WAIT runtime");
}

void Fortran::lower::genLockStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::LockStmt &) {
  TODO(converter.getCurrentLocation(), "LOCK runtime");
}

void Fortran::lower::genUnlockStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::UnlockStmt &) {
  TODO(converter.getCurrentLocation(), "UNLOCK runtime");
}

void Fortran::lower::genSyncAllStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncAllStmt &) {
  TODO(converter.getCurrentLocation(), "SYNC ALL runtime");
}

void Fortran::lower::genSyncImagesStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncImagesStmt &) {
  TODO(converter.getCurrentLocation(), "SYNC IMAGES runtime");
}

void Fortran::lower::genSyncMemoryStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncMemoryStmt &) {
  TODO(converter.getCurrentLocation(), "SYNC MEMORY runtime");
}

void Fortran::lower::genSyncTeamStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncTeamStmt &) {
  TODO(converter.getCurrentLocation(), "SYNC TEAM runtime");
}

void Fortran::lower::genPauseStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::PauseStmt &) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  mlir::func::FuncOp callee =
      fir::runtime::getRuntimeFunc<mkRTKey(PauseStatement)>(loc, builder);
  builder.create<fir::CallOp>(loc, callee, llvm::None);
}

mlir::Value Fortran::lower::genAssociated(fir::FirOpBuilder &builder,
                                          mlir::Location loc,
                                          mlir::Value pointer,
                                          mlir::Value target) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(PointerIsAssociatedWith)>(loc,
                                                                     builder);
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, func.getFunctionType(), pointer, target);
  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

mlir::Value Fortran::lower::genCpuTime(fir::FirOpBuilder &builder,
                                       mlir::Location loc) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(CpuTime)>(loc, builder);
  return builder.create<fir::CallOp>(loc, func, llvm::None).getResult(0);
}

void Fortran::lower::genDateAndTime(fir::FirOpBuilder &builder,
                                    mlir::Location loc,
                                    llvm::Optional<fir::CharBoxValue> date,
                                    llvm::Optional<fir::CharBoxValue> time,
                                    llvm::Optional<fir::CharBoxValue> zone,
                                    mlir::Value values) {
  mlir::func::FuncOp callee =
      fir::runtime::getRuntimeFunc<mkRTKey(DateAndTime)>(loc, builder);
  mlir::FunctionType funcTy = callee.getFunctionType();
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

  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, funcTy.getInput(7));

  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, funcTy, dateBuffer, dateLen, timeBuffer, timeLen,
      zoneBuffer, zoneLen, sourceFile, sourceLine, values);
  builder.create<fir::CallOp>(loc, callee, args);
}

void Fortran::lower::genRandomInit(fir::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Value repeatable,
                                   mlir::Value imageDistinct) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(RandomInit)>(loc, builder);
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, func.getFunctionType(), repeatable, imageDistinct);
  builder.create<fir::CallOp>(loc, func, args);
}

void Fortran::lower::genRandomNumber(fir::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Value harvest) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(RandomNumber)>(loc, builder);
  mlir::FunctionType funcTy = func.getFunctionType();
  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, funcTy.getInput(2));
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, funcTy, harvest, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

void Fortran::lower::genRandomSeed(fir::FirOpBuilder &builder,
                                   mlir::Location loc, int argIndex,
                                   mlir::Value argBox) {
  mlir::func::FuncOp func;
  // argIndex is the nth (0-origin) argument in declaration order,
  // or -1 if no argument is present.
  switch (argIndex) {
  case -1:
    func = fir::runtime::getRuntimeFunc<mkRTKey(RandomSeedDefaultPut)>(loc,
                                                                       builder);
    builder.create<fir::CallOp>(loc, func);
    return;
  case 0:
    func = fir::runtime::getRuntimeFunc<mkRTKey(RandomSeedSize)>(loc, builder);
    break;
  case 1:
    func = fir::runtime::getRuntimeFunc<mkRTKey(RandomSeedPut)>(loc, builder);
    break;
  case 2:
    func = fir::runtime::getRuntimeFunc<mkRTKey(RandomSeedGet)>(loc, builder);
    break;
  default:
    llvm::report_fatal_error("invalid RANDOM_SEED argument index");
  }
  mlir::FunctionType funcTy = func.getFunctionType();
  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, funcTy.getInput(2));
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, funcTy, argBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// generate runtime call to transfer intrinsic with no size argument
void Fortran::lower::genTransfer(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value resultBox, mlir::Value sourceBox,
                                 mlir::Value moldBox) {

  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(Transfer)>(loc, builder);
  mlir::FunctionType fTy = func.getFunctionType();
  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, fTy, resultBox, sourceBox, moldBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// generate runtime call to transfer intrinsic with size argument
void Fortran::lower::genTransferSize(fir::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Value resultBox,
                                     mlir::Value sourceBox, mlir::Value moldBox,
                                     mlir::Value size) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(TransferSize)>(loc, builder);
  mlir::FunctionType fTy = func.getFunctionType();
  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  llvm::SmallVector<mlir::Value> args =
      fir::runtime::createArguments(builder, loc, fTy, resultBox, sourceBox,
                                    moldBox, sourceFile, sourceLine, size);
  builder.create<fir::CallOp>(loc, func, args);
}

/// generate system_clock runtime call/s
/// all intrinsic arguments are optional and may appear here as mlir::Value{}
void Fortran::lower::genSystemClock(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value count,
                                    mlir::Value rate, mlir::Value max) {
  auto makeCall = [&](mlir::func::FuncOp func, mlir::Value arg) {
    mlir::Type type = arg.getType();
    fir::IfOp ifOp{};
    const bool isOptionalArg =
        fir::valueHasFirAttribute(arg, fir::getOptionalAttrName());
    if (type.dyn_cast<fir::PointerType>() || type.dyn_cast<fir::HeapType>()) {
      // Check for a disassociated pointer or an unallocated allocatable.
      assert(!isOptionalArg && "invalid optional argument");
      ifOp = builder.create<fir::IfOp>(loc, builder.genIsNotNullAddr(loc, arg),
                                       /*withElseRegion=*/false);
    } else if (isOptionalArg) {
      ifOp = builder.create<fir::IfOp>(
          loc, builder.create<fir::IsPresentOp>(loc, builder.getI1Type(), arg),
          /*withElseRegion=*/false);
    }
    if (ifOp)
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    mlir::Type kindTy = func.getFunctionType().getInput(0);
    int integerKind = 8;
    if (auto intType = fir::unwrapRefType(type).dyn_cast<mlir::IntegerType>())
      integerKind = intType.getWidth() / 8;
    mlir::Value kind = builder.createIntegerConstant(loc, kindTy, integerKind);
    mlir::Value res =
        builder.create<fir::CallOp>(loc, func, mlir::ValueRange{kind})
            .getResult(0);
    mlir::Value castRes =
        builder.createConvert(loc, fir::dyn_cast_ptrEleTy(type), res);
    builder.create<fir::StoreOp>(loc, castRes, arg);
    if (ifOp)
      builder.setInsertionPointAfter(ifOp);
  };
  using fir::runtime::getRuntimeFunc;
  if (count)
    makeCall(getRuntimeFunc<mkRTKey(SystemClockCount)>(loc, builder), count);
  if (rate)
    makeCall(getRuntimeFunc<mkRTKey(SystemClockCountRate)>(loc, builder), rate);
  if (max)
    makeCall(getRuntimeFunc<mkRTKey(SystemClockCountMax)>(loc, builder), max);
}
