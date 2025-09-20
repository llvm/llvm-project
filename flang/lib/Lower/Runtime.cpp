//===-- Runtime.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Runtime.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/OpenACC.h"
#include "flang/Lower/OpenMP.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/Coarray.h"
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
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "flang-lower-runtime"

using namespace Fortran::runtime;

/// Runtime calls that do not return to the caller indicate this condition by
/// terminating the current basic block with an unreachable op.
static void genUnreachable(fir::FirOpBuilder &builder, mlir::Location loc) {
  mlir::Block *curBlock = builder.getBlock();
  mlir::Operation *parentOp = curBlock->getParentOp();
  if (parentOp->getDialect()->getNamespace() ==
      mlir::omp::OpenMPDialect::getDialectNamespace())
    Fortran::lower::genOpenMPTerminator(builder, parentOp, loc);
  else if (Fortran::lower::isInsideOpenACCComputeConstruct(builder))
    Fortran::lower::genOpenACCTerminator(builder, parentOp, loc);
  else
    fir::UnreachableOp::create(builder, loc);
  mlir::Block *newBlock = curBlock->splitBlock(builder.getInsertionPoint());
  builder.setInsertionPointToStart(newBlock);
}

/// Initializes values for STAT and ERRMSG
static std::pair<mlir::Value, mlir::Value> getStatAndErrmsg(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const std::list<Fortran::parser::StatOrErrmsg> &statOrErrList) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;

  mlir::Value errMsgExpr, statExpr;
  for (const Fortran::parser::StatOrErrmsg &statOrErr : statOrErrList) {
    std::visit(Fortran::common::visitors{
                   [&](const Fortran::parser::StatVariable &statVar) {
                     statExpr = fir::getBase(converter.genExprAddr(
                         loc, Fortran::semantics::GetExpr(statVar), stmtCtx));
                   },
                   [&](const Fortran::parser::MsgVariable &errMsgVar) {
                     const Fortran::semantics::SomeExpr *expr =
                         Fortran::semantics::GetExpr(errMsgVar);
                     errMsgExpr = fir::getBase(
                         converter.genExprBox(loc, *expr, stmtCtx));
                   }},
               statOrErr.u);
  }

  if (!statExpr) {
    statExpr = fir::AbsentOp::create(builder, loc,
                                     builder.getRefType(builder.getI32Type()));
  }
  if (!errMsgExpr) {
    errMsgExpr = fir::AbsentOp::create(
        builder, loc,
        fir::BoxType::get(fir::CharacterType::get(
            builder.getContext(), 1, fir::CharacterType::unknownLen())));
  }
  return {statExpr, errMsgExpr};
}

//===----------------------------------------------------------------------===//
// Misc. Fortran statements that lower to runtime calls
//===----------------------------------------------------------------------===//

void Fortran::lower::genStopStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::StopStmt &stmt) {
  const bool isError = std::get<Fortran::parser::StopStmt::Kind>(stmt.t) ==
                       Fortran::parser::StopStmt::Kind::ErrorStop;
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
    // Default to values are advised in F'2023 11.4 p2.
    operands.push_back(builder.createIntegerConstant(
        loc, calleeType.getInput(0), isError ? 1 : 0));
  }

  // Second operand indicates ERROR STOP
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

  fir::CallOp::create(builder, loc, callee, operands);
  auto blockIsUnterminated = [&builder]() {
    mlir::Block *currentBlock = builder.getBlock();
    return currentBlock->empty() ||
           !currentBlock->back().hasTrait<mlir::OpTrait::IsTerminator>();
  };
  if (blockIsUnterminated())
    genUnreachable(builder, loc);
}

void Fortran::lower::genFailImageStatement(
    Fortran::lower::AbstractConverter &converter) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  mlir::func::FuncOp callee =
      fir::runtime::getRuntimeFunc<mkRTKey(FailImageStatement)>(loc, builder);
  fir::CallOp::create(builder, loc, callee, mlir::ValueRange{});
  genUnreachable(builder, loc);
}

void Fortran::lower::genNotifyWaitStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::NotifyWaitStmt &) {
  TODO(converter.getCurrentLocation(), "coarray: NOTIFY WAIT runtime");
}

void Fortran::lower::genEventPostStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::EventPostStmt &) {
  TODO(converter.getCurrentLocation(), "coarray: EVENT POST runtime");
}

void Fortran::lower::genEventWaitStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::EventWaitStmt &) {
  TODO(converter.getCurrentLocation(), "coarray: EVENT WAIT runtime");
}

void Fortran::lower::genLockStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::LockStmt &) {
  TODO(converter.getCurrentLocation(), "coarray: LOCK runtime");
}

void Fortran::lower::genUnlockStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::UnlockStmt &) {
  TODO(converter.getCurrentLocation(), "coarray: UNLOCK runtime");
}

void Fortran::lower::genSyncAllStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncAllStmt &stmt) {
  mlir::Location loc = converter.getCurrentLocation();
  converter.checkCoarrayEnabled();

  // Handle STAT and ERRMSG values
  const std::list<Fortran::parser::StatOrErrmsg> &statOrErrList = stmt.v;
  auto [statAddr, errMsgAddr] = getStatAndErrmsg(converter, loc, statOrErrList);

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  fir::runtime::genSyncAllStatement(builder, loc, statAddr, errMsgAddr);
}

void Fortran::lower::genSyncImagesStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncImagesStmt &stmt) {
  mlir::Location loc = converter.getCurrentLocation();
  converter.checkCoarrayEnabled();
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  // Handle STAT and ERRMSG values
  const std::list<Fortran::parser::StatOrErrmsg> &statOrErrList =
      std::get<std::list<Fortran::parser::StatOrErrmsg>>(stmt.t);
  auto [statAddr, errMsgAddr] = getStatAndErrmsg(converter, loc, statOrErrList);

  // SYNC_IMAGES(*) is passed as count == -1 while  SYNC IMAGES([]) has count
  // == 0. Note further that SYNC IMAGES(*) is not semantically equivalent to
  // SYNC ALL.
  Fortran::lower::StatementContext stmtCtx;
  mlir::Value imageSet;
  const Fortran::parser::SyncImagesStmt::ImageSet &imgSet =
      std::get<Fortran::parser::SyncImagesStmt::ImageSet>(stmt.t);
  std::visit(Fortran::common::visitors{
                 [&](const Fortran::parser::IntExpr &intExpr) {
                   const SomeExpr *expr = Fortran::semantics::GetExpr(intExpr);
                   imageSet =
                       fir::getBase(converter.genExprBox(loc, *expr, stmtCtx));
                 },
                 [&](const Fortran::parser::Star &) {
                   imageSet = fir::AbsentOp::create(
                       builder, loc,
                       fir::BoxType::get(fir::SequenceType::get(
                           {fir::SequenceType::getUnknownExtent()},
                           builder.getI32Type())));
                 }},
             imgSet.u);

  fir::runtime::genSyncImagesStatement(builder, loc, imageSet, statAddr,
                                       errMsgAddr);
}

void Fortran::lower::genSyncMemoryStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncMemoryStmt &stmt) {
  mlir::Location loc = converter.getCurrentLocation();
  converter.checkCoarrayEnabled();

  // Handle STAT and ERRMSG values
  const std::list<Fortran::parser::StatOrErrmsg> &statOrErrList = stmt.v;
  auto [statAddr, errMsgAddr] = getStatAndErrmsg(converter, loc, statOrErrList);

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  fir::runtime::genSyncMemoryStatement(builder, loc, statAddr, errMsgAddr);
}

void Fortran::lower::genSyncTeamStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncTeamStmt &) {
  TODO(converter.getCurrentLocation(), "coarray: SYNC TEAM runtime");
}

void Fortran::lower::genPauseStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::PauseStmt &) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  mlir::func::FuncOp callee =
      fir::runtime::getRuntimeFunc<mkRTKey(PauseStatement)>(loc, builder);
  fir::CallOp::create(builder, loc, callee, mlir::ValueRange{});
}

void Fortran::lower::genPointerAssociate(fir::FirOpBuilder &builder,
                                         mlir::Location loc,
                                         mlir::Value pointer,
                                         mlir::Value target) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(PointerAssociate)>(loc, builder);
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, func.getFunctionType(), pointer, target);
  fir::CallOp::create(builder, loc, func, args);
}

void Fortran::lower::genPointerAssociateRemapping(
    fir::FirOpBuilder &builder, mlir::Location loc, mlir::Value pointer,
    mlir::Value target, mlir::Value bounds, bool isMonomorphic) {
  mlir::func::FuncOp func =
      isMonomorphic
          ? fir::runtime::getRuntimeFunc<mkRTKey(
                PointerAssociateRemappingMonomorphic)>(loc, builder)
          : fir::runtime::getRuntimeFunc<mkRTKey(PointerAssociateRemapping)>(
                loc, builder);
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, func.getFunctionType(), pointer, target, bounds, sourceFile,
      sourceLine);
  fir::CallOp::create(builder, loc, func, args);
}

void Fortran::lower::genPointerAssociateLowerBounds(fir::FirOpBuilder &builder,
                                                    mlir::Location loc,
                                                    mlir::Value pointer,
                                                    mlir::Value target,
                                                    mlir::Value lbounds) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(PointerAssociateLowerBounds)>(
          loc, builder);
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, func.getFunctionType(), pointer, target, lbounds);
  fir::CallOp::create(builder, loc, func, args);
}
