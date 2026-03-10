//===-- MultiImageFortran.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Implementation of the lowering of image related constructs and expressions.
/// Fortran images can form teams, communicate via coarrays, etc.
///
//===----------------------------------------------------------------------===//

#include "flang/Lower/MultiImageFortran.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"

//===----------------------------------------------------------------------===//
// Synchronization statements
//===----------------------------------------------------------------------===//

void Fortran::lower::genSyncAllStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncAllStmt &stmt) {
  mlir::Location loc = converter.getCurrentLocation();
  converter.checkCoarrayEnabled();

  // Handle STAT and ERRMSG values
  const std::list<Fortran::parser::StatOrErrmsg> &statOrErrList = stmt.v;
  auto [statAddr, errMsgAddr] = converter.genStatAndErrmsg(loc, statOrErrList);

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mif::SyncAllOp::create(builder, loc, statAddr, errMsgAddr);
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
  auto [statAddr, errMsgAddr] = converter.genStatAndErrmsg(loc, statOrErrList);

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
                   // Image set is not set.
                   imageSet = mlir::Value{};
                 }},
             imgSet.u);

  mif::SyncImagesOp::create(builder, loc, imageSet, statAddr, errMsgAddr);
}

void Fortran::lower::genSyncMemoryStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncMemoryStmt &stmt) {
  mlir::Location loc = converter.getCurrentLocation();
  converter.checkCoarrayEnabled();

  // Handle STAT and ERRMSG values
  const std::list<Fortran::parser::StatOrErrmsg> &statOrErrList = stmt.v;
  auto [statAddr, errMsgAddr] = converter.genStatAndErrmsg(loc, statOrErrList);

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mif::SyncMemoryOp::create(builder, loc, statAddr, errMsgAddr);
}

void Fortran::lower::genSyncTeamStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncTeamStmt &stmt) {
  mlir::Location loc = converter.getCurrentLocation();
  converter.checkCoarrayEnabled();

  // Handle TEAM
  Fortran::lower::StatementContext stmtCtx;
  const Fortran::parser::TeamValue &teamValue =
      std::get<Fortran::parser::TeamValue>(stmt.t);
  const SomeExpr *teamExpr = Fortran::semantics::GetExpr(teamValue);
  mlir::Value team =
      fir::getBase(converter.genExprBox(loc, *teamExpr, stmtCtx));

  // Handle STAT and ERRMSG values
  const std::list<Fortran::parser::StatOrErrmsg> &statOrErrList =
      std::get<std::list<Fortran::parser::StatOrErrmsg>>(stmt.t);
  auto [statAddr, errMsgAddr] = converter.genStatAndErrmsg(loc, statOrErrList);

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mif::SyncTeamOp::create(builder, loc, team, statAddr, errMsgAddr);
}

//===----------------------------------------------------------------------===//
// TEAM statements and constructs
//===----------------------------------------------------------------------===//

void Fortran::lower::genChangeTeamConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &,
    const Fortran::parser::ChangeTeamConstruct &) {
  TODO(converter.getCurrentLocation(), "coarray: CHANGE TEAM construct");
}

mif::ChangeTeamOp
Fortran::lower::genChangeTeamStmt(Fortran::lower::AbstractConverter &converter,
                                  Fortran::lower::pft::Evaluation &,
                                  const Fortran::parser::ChangeTeamStmt &stmt) {
  mlir::Location loc = converter.getCurrentLocation();
  converter.checkCoarrayEnabled();
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  mlir::Value errMsgAddr, statAddr, team;
  // Handle STAT and ERRMSG values
  Fortran::lower::StatementContext stmtCtx;
  const std::list<Fortran::parser::StatOrErrmsg> &statOrErrList =
      std::get<std::list<Fortran::parser::StatOrErrmsg>>(stmt.t);
  for (const Fortran::parser::StatOrErrmsg &statOrErr : statOrErrList) {
    std::visit(Fortran::common::visitors{
                   [&](const Fortran::parser::StatVariable &statVar) {
                     const auto *expr = Fortran::semantics::GetExpr(statVar);
                     statAddr = fir::getBase(
                         converter.genExprAddr(loc, *expr, stmtCtx));
                   },
                   [&](const Fortran::parser::MsgVariable &errMsgVar) {
                     const auto *expr = Fortran::semantics::GetExpr(errMsgVar);
                     errMsgAddr = fir::getBase(
                         converter.genExprBox(loc, *expr, stmtCtx));
                   },
               },
               statOrErr.u);
  }

  // TODO: Manage the list of coarrays associated in
  // `std::list<CoarrayAssociation>`. According to the PRIF specification, it is
  // necessary to call `prif_alias_{create|destroy}` for each coarray defined in
  // this list. Support will be added once lowering to this procedure is
  // possible.
  const std::list<Fortran::parser::CoarrayAssociation> &coarrayAssocList =
      std::get<std::list<Fortran::parser::CoarrayAssociation>>(stmt.t);
  if (coarrayAssocList.size())
    TODO(loc, "Coarrays provided in the association list.");

  // Handle TEAM-VALUE
  const auto *teamExpr =
      Fortran::semantics::GetExpr(std::get<Fortran::parser::TeamValue>(stmt.t));
  team = fir::getBase(converter.genExprBox(loc, *teamExpr, stmtCtx));

  return mif::ChangeTeamOp::create(builder, loc, team, statAddr, errMsgAddr);
}

void Fortran::lower::genEndChangeTeamStmt(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &,
    const Fortran::parser::EndChangeTeamStmt &stmt) {
  converter.checkCoarrayEnabled();
  mlir::Location loc = converter.getCurrentLocation();
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  mlir::Value errMsgAddr, statAddr;
  // Handle STAT and ERRMSG values
  Fortran::lower::StatementContext stmtCtx;
  const std::list<Fortran::parser::StatOrErrmsg> &statOrErrList =
      std::get<std::list<Fortran::parser::StatOrErrmsg>>(stmt.t);
  for (const Fortran::parser::StatOrErrmsg &statOrErr : statOrErrList) {
    std::visit(Fortran::common::visitors{
                   [&](const Fortran::parser::StatVariable &statVar) {
                     const auto *expr = Fortran::semantics::GetExpr(statVar);
                     statAddr = fir::getBase(
                         converter.genExprAddr(loc, *expr, stmtCtx));
                   },
                   [&](const Fortran::parser::MsgVariable &errMsgVar) {
                     const auto *expr = Fortran::semantics::GetExpr(errMsgVar);
                     errMsgAddr = fir::getBase(
                         converter.genExprBox(loc, *expr, stmtCtx));
                   },
               },
               statOrErr.u);
  }

  mif::EndTeamOp::create(builder, loc, statAddr, errMsgAddr);
}

void Fortran::lower::genFormTeamStatement(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &,
    const Fortran::parser::FormTeamStmt &stmt) {
  converter.checkCoarrayEnabled();
  mlir::Location loc = converter.getCurrentLocation();
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  mlir::Value errMsgAddr, statAddr, newIndex, teamNumber, team;
  // Handle NEW_INDEX, STAT and ERRMSG
  std::list<Fortran::parser::StatOrErrmsg> statOrErrList{};
  Fortran::lower::StatementContext stmtCtx;
  const auto &formSpecList =
      std::get<std::list<Fortran::parser::FormTeamStmt::FormTeamSpec>>(stmt.t);
  for (const Fortran::parser::FormTeamStmt::FormTeamSpec &formSpec :
       formSpecList) {
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::StatOrErrmsg &statOrErr) {
              std::visit(
                  Fortran::common::visitors{
                      [&](const Fortran::parser::StatVariable &statVar) {
                        const auto *expr = Fortran::semantics::GetExpr(statVar);
                        statAddr = fir::getBase(
                            converter.genExprAddr(loc, *expr, stmtCtx));
                      },
                      [&](const Fortran::parser::MsgVariable &errMsgVar) {
                        const auto *expr =
                            Fortran::semantics::GetExpr(errMsgVar);
                        errMsgAddr = fir::getBase(
                            converter.genExprBox(loc, *expr, stmtCtx));
                      },
                  },
                  statOrErr.u);
            },
            [&](const Fortran::parser::ScalarIntExpr &intExpr) {
              fir::ExtendedValue newIndexExpr = converter.genExprValue(
                  loc, Fortran::semantics::GetExpr(intExpr), stmtCtx);
              newIndex = fir::getBase(newIndexExpr);
            },
        },
        formSpec.u);
  }

  // Handle TEAM-NUMBER
  const auto *teamNumberExpr = Fortran::semantics::GetExpr(
      std::get<Fortran::parser::ScalarIntExpr>(stmt.t));
  teamNumber =
      fir::getBase(converter.genExprValue(loc, *teamNumberExpr, stmtCtx));

  // Handle TEAM-VARIABLE
  const auto *teamExpr = Fortran::semantics::GetExpr(
      std::get<Fortran::parser::TeamVariable>(stmt.t));
  team = fir::getBase(converter.genExprBox(loc, *teamExpr, stmtCtx));

  mif::FormTeamOp::create(builder, loc, teamNumber, team, newIndex, statAddr,
                          errMsgAddr);
}

//===----------------------------------------------------------------------===//
// COARRAY expressions
//===----------------------------------------------------------------------===//

fir::ExtendedValue Fortran::lower::CoarrayExprHelper::genAddr(
    const Fortran::evaluate::CoarrayRef &expr) {
  (void)symMap;
  TODO(converter.getCurrentLocation(), "co-array address");
}

fir::ExtendedValue Fortran::lower::CoarrayExprHelper::genValue(
    const Fortran::evaluate::CoarrayRef &expr) {
  TODO(converter.getCurrentLocation(), "co-array value");
}
