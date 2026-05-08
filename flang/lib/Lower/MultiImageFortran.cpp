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
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/MIFCommon.h"
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
// COARRAY utils
//===----------------------------------------------------------------------===//

mlir::Value
Fortran::lower::genLowerCoBounds(Fortran::lower::AbstractConverter &converter,
                                 mlir::Location loc,
                                 const Fortran::semantics::Symbol &sym) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  mlir::Type i64Ty = builder.getI64Type();
  mlir::Type addrType = builder.getRefType(i64Ty);
  mlir::Value one = builder.createIntegerConstant(loc, i64Ty, 1);
  mlir::Value lcobounds;

  if (Fortran::semantics::IsAllocatableOrObjectPointer(&sym))
    mlir::emitError(
        loc,
        "Unable to use genLowerCoBounds on ALLOCATABLE and POINTER symbol");
  if (const auto *object =
          sym.GetUltimate()
              .detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
    size_t corank = object->coshape().size();
    mlir::Type arrayType = fir::SequenceType::get(
        {static_cast<fir::SequenceType::Extent>(corank)}, i64Ty);
    lcobounds = builder.createTemporary(loc, arrayType);
    mlir::Value lcovalue = one; // default lcobounds
    for (size_t i = 0; i < corank; i++) {
      if (auto lb = object->coshape()[i].lbound().GetExplicit()) {
        auto lbExpr = ignoreEvConvert(*lb);
        lcovalue = fir::getBase(converter.genExprValue(loc, lbExpr, stmtCtx));
      }

      if (lcovalue.getType() != i64Ty)
        lcovalue = fir::ConvertOp::create(builder, loc, i64Ty, lcovalue);
      mlir::Value index =
          builder.createIntegerConstant(loc, builder.getIndexType(), i);
      mlir::Value lcoaddr =
          fir::CoordinateOp::create(builder, loc, addrType, lcobounds, index);
      fir::StoreOp::create(builder, loc, lcovalue, lcoaddr);
    }
    lcobounds = builder.createBox(loc, lcobounds);
  }
  return lcobounds;
}

mlir::Value
Fortran::lower::genUpperCoBounds(Fortran::lower::AbstractConverter &converter,
                                 mlir::Location loc,
                                 const Fortran::semantics::Symbol &sym) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  mlir::Type i64Ty = builder.getI64Type();
  mlir::Type addrType = builder.getRefType(i64Ty);
  mlir::Value one = builder.createIntegerConstant(loc, i64Ty, 1);
  mlir::Value ucobounds;

  if (Fortran::semantics::IsAllocatableOrObjectPointer(&sym))
    mlir::emitError(
        loc,
        "Unable to use genUpperCoBounds on ALLOCATABLE and POINTER symbol");
  if (const auto *object =
          sym.GetUltimate()
              .detailsIf<Fortran::semantics::ObjectEntityDetails>()) {
    size_t corank = object->coshape().size();
    // PRIF take an array of size corank-1 for ucobound.
    mlir::Type arrayType = fir::SequenceType::get(
        {static_cast<fir::SequenceType::Extent>(corank - 1)}, i64Ty);
    ucobounds = builder.createTemporary(loc, arrayType);
    mlir::Value ucovalue;
    for (size_t i = 0; i < corank - 1; i++) {
      if (auto ub = object->coshape()[i].lbound().GetExplicit()) {
        auto ubExpr = ignoreEvConvert(*ub);
        ucovalue = fir::getBase(converter.genExprValue(loc, ubExpr, stmtCtx));
      } else {
        if (auto lb = object->coshape()[i].lbound().GetExplicit()) {
          auto lbExpr = ignoreEvConvert(*lb);
          ucovalue = fir::getBase(converter.genExprValue(
              loc, lbExpr, stmtCtx)); // default value from lcobound
        } else
          ucovalue = one;
      }
      if (ucovalue.getType() != i64Ty)
        ucovalue = fir::ConvertOp::create(builder, loc, i64Ty, ucovalue);
      mlir::Value index =
          builder.createIntegerConstant(loc, builder.getIndexType(), i);
      mlir::Value ucoaddr =
          fir::CoordinateOp::create(builder, loc, addrType, ucobounds, index);
      fir::StoreOp::create(builder, loc, ucovalue, ucoaddr);
    }
    ucobounds = builder.createBox(loc, ucobounds);
  }
  return ucobounds;
}

static std::tuple<mlir::Value, mlir::Value>
genCoBounds(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
            const Fortran::parser::AllocateCoarraySpec &allocSpec) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  llvm::SmallVector<std::int64_t> lcbs, ucbs;
  mlir::Type i64Ty = builder.getI64Type();
  mlir::Type addrType = builder.getRefType(i64Ty);
  mlir::Value one = builder.createIntegerConstant(loc, i64Ty, 1);

  const std::list<Fortran::parser::AllocateCoshapeSpec> &coshapeSpecs =
      std::get<0>(allocSpec.t);
  size_t corank = coshapeSpecs.size() + 1;
  mlir::Type arrayType = fir::SequenceType::get(
      {static_cast<fir::SequenceType::Extent>(corank)}, i64Ty);
  mlir::Value lcobounds = builder.createTemporary(loc, arrayType);
  mlir::Type arrayType2 = fir::SequenceType::get(
      {static_cast<fir::SequenceType::Extent>(corank - 1)}, i64Ty);
  mlir::Value ucobounds = builder.createTemporary(loc, arrayType2);
  size_t i = 0;
  for (const Fortran::parser::AllocateCoshapeSpec &coshapeSpec : coshapeSpecs) {
    const std::optional<Fortran::parser::BoundExpr> &lbExpr =
        std::get<0>(coshapeSpec.t);
    const Fortran::parser::BoundExpr &ubExpr = std::get<1>(coshapeSpec.t);
    mlir::Value lb = one;
    if (lbExpr.has_value()) {
      auto expr = Fortran::semantics::GetExpr(*lbExpr);
      lb = fir::getBase(converter.genExprValue(loc, expr, stmtCtx));
      if (lb.getType() != i64Ty)
        lb = fir::ConvertOp::create(builder, loc, i64Ty, lb);
    }

    auto ube = Fortran::semantics::GetExpr(ubExpr);
    mlir::Value ub = fir::getBase(converter.genExprValue(loc, ube, stmtCtx));
    if (ub.getType() != i64Ty)
      ub = fir::ConvertOp::create(builder, loc, i64Ty, ub);

    mlir::Value index =
        builder.createIntegerConstant(loc, builder.getIndexType(), i);
    // Lcobound
    mlir::Value lcoaddr =
        fir::CoordinateOp::create(builder, loc, addrType, lcobounds, index);
    fir::StoreOp::create(builder, loc, lb, lcoaddr);
    // Ucobound
    mlir::Value ucoaddr =
        fir::CoordinateOp::create(builder, loc, addrType, ucobounds, index);
    fir::StoreOp::create(builder, loc, ub, ucoaddr);
    i++;
  }
  // Last lcobound;
  {
    mlir::Value lb = one;
    if (const std::optional<Fortran::parser::BoundExpr> &lastCobound =
            std::get<1>(allocSpec.t)) {
      auto expr = Fortran::semantics::GetExpr(*lastCobound);
      lb = fir::getBase(converter.genExprValue(loc, expr, stmtCtx));
    }
    mlir::Value index =
        builder.createIntegerConstant(loc, builder.getIndexType(), i);
    mlir::Value lcoaddr =
        fir::CoordinateOp::create(builder, loc, addrType, lcobounds, index);
    fir::StoreOp::create(builder, loc, lb, lcoaddr);
  }

  lcobounds = builder.createBox(loc, lcobounds);
  ucobounds = builder.createBox(loc, ucobounds);
  return {lcobounds, ucobounds};
}

mlir::Value Fortran::lower::genAllocateCoarray(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::semantics::Symbol &sym, mlir::Value addr,
    const std::optional<Fortran::parser::AllocateCoarraySpec> &allocSpec,
    mlir::Value errmsg, bool hasStat) {
  converter.checkCoarrayEnabled();
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  mlir::Value stat;
  if (hasStat)
    stat = builder.createTemporary(loc, builder.getI32Type());

  mlir::Value lcobounds, ucobounds;
  if (allocSpec.has_value()) {
    std::tie(lcobounds, ucobounds) = genCoBounds(converter, loc, *allocSpec);
  } else {
    lcobounds = Fortran::lower::genLowerCoBounds(converter, loc, sym);
    ucobounds = Fortran::lower::genUpperCoBounds(converter, loc, sym);
  }
  std::string uniqName = mif::getFullUniqName(addr);
  if (uniqName.empty())
    uniqName = converter.mangleName(sym);
  mif::AllocCoarrayOp::create(builder, loc, addr, uniqName, lcobounds,
                              ucobounds, stat, errmsg);
  return stat;
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
