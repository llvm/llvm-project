//===-- OpenACC.cpp -- OpenACC directive lowering -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/OpenACC.h"
#include "flang/Common/idioms.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "llvm/Frontend/OpenACC/ACC.h.inc"

#define TODO() llvm_unreachable("not yet implemented")

static const Fortran::parser::Name *
getDesignatorNameIfDataRef(const Fortran::parser::Designator &designator) {
  const auto *dataRef{std::get_if<Fortran::parser::DataRef>(&designator.u)};
  return dataRef ? std::get_if<Fortran::parser::Name>(&dataRef->u) : nullptr;
}

static void genObjectList(const Fortran::parser::AccObjectList &objectList,
                          Fortran::lower::AbstractConverter &converter,
                          std::int32_t &objectsCount,
                          SmallVector<Value, 8> &operands) {
  for (const auto &accObject : objectList.v) {
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::Designator &designator) {
              if (const auto *name = getDesignatorNameIfDataRef(designator)) {
                ++objectsCount;
                const auto variable = converter.getSymbolAddress(*name->symbol);
                operands.push_back(variable);
              }
            },
            [&](const Fortran::parser::Name &name) {
              ++objectsCount;
              const auto variable = converter.getSymbolAddress(*name.symbol);
              operands.push_back(variable);
            }},
        accObject.u);
  }
}

static void genACC(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenACCLoopConstruct &loopConstruct) {

  const auto &beginLoopDirective =
      std::get<Fortran::parser::AccBeginLoopDirective>(loopConstruct.t);
  const auto &loopDirective =
      std::get<Fortran::parser::AccLoopDirective>(beginLoopDirective.t);

  if (loopDirective.v == llvm::acc::ACCD_loop) {
    auto &firOpBuilder = converter.getFirOpBuilder();
    auto currentLocation = converter.getCurrentLocation();
    llvm::ArrayRef<mlir::Type> argTy;

    // Add attribute extracted from clauses.
    const auto &accClauseList =
        std::get<Fortran::parser::AccClauseList>(beginLoopDirective.t);

    mlir::Value workerNum;
    mlir::Value vectorLength;
    mlir::Value gangNum;
    mlir::Value gangStatic;
    std::int32_t tileOperands = 0;
    std::int32_t privateOperands = 0;
    std::int32_t reductionOperands = 0;
    std::int64_t executionMapping = mlir::acc::OpenACCExecMapping::NONE;
    SmallVector<Value, 8> operands;

    // Lower clauses values mapped to operands.
    for (const auto &clause : accClauseList.v) {
      if (const auto *gangClause =
              std::get_if<Fortran::parser::AccClause::Gang>(&clause.u)) {
        if (gangClause->v) {
          const Fortran::parser::AccGangArgument &x = *gangClause->v;
          if (const auto &gangNumValue =
                  std::get<std::optional<Fortran::parser::ScalarIntExpr>>(
                      x.t)) {
            gangNum = converter.genExprValue(
                *Fortran::semantics::GetExpr(gangNumValue.value()));
            operands.push_back(gangNum);
          }
          if (const auto &gangStaticValue =
                  std::get<std::optional<Fortran::parser::AccSizeExpr>>(x.t)) {
            const auto &expr =
                std::get<std::optional<Fortran::parser::ScalarIntExpr>>(
                    gangStaticValue.value().t);
            if (expr) {
              gangStatic =
                  converter.genExprValue(*Fortran::semantics::GetExpr(*expr));
            } else {
              // * was passed as value and will be represented as a -1 constant
              // integer.
              gangStatic = firOpBuilder.createIntegerConstant(
                  currentLocation, firOpBuilder.getIntegerType(32),
                  /* STAR */ -1);
            }
            operands.push_back(gangStatic);
          }
        }
        executionMapping |= mlir::acc::OpenACCExecMapping::GANG;
      } else if (const auto *workerClause =
                     std::get_if<Fortran::parser::AccClause::Worker>(
                         &clause.u)) {
        if (workerClause->v) {
          workerNum = converter.genExprValue(
              *Fortran::semantics::GetExpr(*workerClause->v));
          operands.push_back(workerNum);
        }
        executionMapping |= mlir::acc::OpenACCExecMapping::WORKER;
      } else if (const auto *vectorClause =
                     std::get_if<Fortran::parser::AccClause::Vector>(
                         &clause.u)) {
        if (vectorClause->v) {
          vectorLength = converter.genExprValue(
              *Fortran::semantics::GetExpr(*vectorClause->v));
          operands.push_back(vectorLength);
        }
        executionMapping |= mlir::acc::OpenACCExecMapping::VECTOR;
      } else if (const auto *tileClause =
                     std::get_if<Fortran::parser::AccClause::Tile>(&clause.u)) {
        const Fortran::parser::AccTileExprList &accTileExprList = tileClause->v;
        for (const auto &accTileExpr : accTileExprList.v) {
          const auto &expr =
              std::get<std::optional<Fortran::parser::ScalarIntConstantExpr>>(
                  accTileExpr.t);
          ++tileOperands;
          if (expr) {
            operands.push_back(
                converter.genExprValue(*Fortran::semantics::GetExpr(*expr)));
          } else {
            // * was passed as value and will be represented as a -1 constant
            // integer.
            mlir::Value tileStar = firOpBuilder.createIntegerConstant(
                currentLocation, firOpBuilder.getIntegerType(32),
                /* STAR */ -1);
            operands.push_back(tileStar);
          }
        }
      } else if (const auto *privateClause =
                     std::get_if<Fortran::parser::AccClause::Private>(
                         &clause.u)) {
        const Fortran::parser::AccObjectList &accObjectList = privateClause->v;
        genObjectList(accObjectList, converter, privateOperands, operands);
      }
      // Reduction clause is left out for the moment as the clause will probably
      // end up having its own operation.
    }

    auto loopOp = firOpBuilder.create<mlir::acc::LoopOp>(currentLocation, argTy,
                                                         operands);

    firOpBuilder.createBlock(&loopOp.getRegion());
    auto &block = loopOp.getRegion().back();
    firOpBuilder.setInsertionPointToStart(&block);
    // ensure the block is well-formed.
    firOpBuilder.create<mlir::acc::YieldOp>(currentLocation);

    loopOp.setAttr(mlir::acc::LoopOp::getOperandSegmentSizeAttr(),
                   firOpBuilder.getI32VectorAttr(
                       {gangNum ? 1 : 0, gangStatic ? 1 : 0, workerNum ? 1 : 0,
                        vectorLength ? 1 : 0, tileOperands, privateOperands,
                        reductionOperands}));

    loopOp.setAttr(mlir::acc::LoopOp::getExecutionMappingAttrName(),
                   firOpBuilder.getI64IntegerAttr(executionMapping));

    // Lower clauses mapped to attributes
    for (const auto &clause : accClauseList.v) {
      if (const auto *collapseClause =
              std::get_if<Fortran::parser::AccClause::Collapse>(&clause.u)) {
        const auto *expr = Fortran::semantics::GetExpr(collapseClause->v);
        const auto collapseValue = Fortran::evaluate::ToInt64(*expr);
        if (collapseValue) {
          loopOp.setAttr(mlir::acc::LoopOp::getCollapseAttrName(),
                         firOpBuilder.getI64IntegerAttr(*collapseValue));
        }
      } else if (std::get_if<Fortran::parser::AccClause::Seq>(&clause.u)) {
        loopOp.setAttr(mlir::acc::LoopOp::getSeqAttrName(),
                       firOpBuilder.getUnitAttr());
      } else if (std::get_if<Fortran::parser::AccClause::Independent>(
                     &clause.u)) {
        loopOp.setAttr(mlir::acc::LoopOp::getIndependentAttrName(),
                       firOpBuilder.getUnitAttr());
      } else if (std::get_if<Fortran::parser::AccClause::Auto>(&clause.u)) {
        loopOp.setAttr(mlir::acc::LoopOp::getAutoAttrName(),
                       firOpBuilder.getUnitAttr());
      }
    }

    // Place the insertion point to the start of the first block.
    firOpBuilder.setInsertionPointToStart(&block);
  }
}

void Fortran::lower::genOpenACCConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenACCConstruct &accConstruct) {

  std::visit(
      common::visitors{
          [&](const Fortran::parser::OpenACCBlockConstruct &blockConstruct) {
            TODO();
          },
          [&](const Fortran::parser::OpenACCCombinedConstruct
                  &combinedConstruct) { TODO(); },
          [&](const Fortran::parser::OpenACCLoopConstruct &loopConstruct) {
            genACC(converter, eval, loopConstruct);
          },
          [&](const Fortran::parser::OpenACCStandaloneConstruct
                  &standaloneConstruct) { TODO(); },
          [&](const Fortran::parser::OpenACCRoutineConstruct
                  &routineConstruct) { TODO(); },
          [&](const Fortran::parser::OpenACCCacheConstruct &cacheConstruct) {
            TODO();
          },
          [&](const Fortran::parser::OpenACCWaitConstruct &waitConstruct) {
            TODO();
          },
          [&](const Fortran::parser::OpenACCAtomicConstruct &atomicConstruct) {
            TODO();
          },
      },
      accConstruct.u);
}
