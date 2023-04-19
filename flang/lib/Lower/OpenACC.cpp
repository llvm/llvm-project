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
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "llvm/Frontend/OpenACC/ACC.h.inc"

// Special value for * passed in device_type or gang clauses.
static constexpr std::int64_t starCst = -1;

static const Fortran::parser::Name *
getDesignatorNameIfDataRef(const Fortran::parser::Designator &designator) {
  const auto *dataRef = std::get_if<Fortran::parser::DataRef>(&designator.u);
  return dataRef ? std::get_if<Fortran::parser::Name>(&dataRef->u) : nullptr;
}

static void
genObjectList(const Fortran::parser::AccObjectList &objectList,
              Fortran::lower::AbstractConverter &converter,
              Fortran::semantics::SemanticsContext &semanticsContext,
              Fortran::lower::StatementContext &stmtCtx,
              llvm::SmallVectorImpl<mlir::Value> &operands) {
  auto addOperands = [&](Fortran::lower::SymbolRef sym) {
    const auto variable = converter.getSymbolAddress(sym);
    // TODO: Might need revisiting to handle for non-shared clauses
    if (variable) {
      operands.push_back(variable);
    } else {
      if (const auto *details =
              sym->detailsIf<Fortran::semantics::HostAssocDetails>())
        operands.push_back(converter.getSymbolAddress(details->symbol()));
    }
  };

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  for (const auto &accObject : objectList.v) {
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::Designator &designator) {
              mlir::Location operandLocation =
                  converter.genLocation(designator.source);
              if (auto expr{Fortran::semantics::AnalyzeExpr(semanticsContext,
                                                            designator)}) {
                if ((*expr).Rank() > 0 &&
                    Fortran::parser::Unwrap<Fortran::parser::ArrayElement>(
                        designator)) {
                  // Array sections.
                  fir::ExtendedValue exV =
                      converter.genExprBox(operandLocation, *expr, stmtCtx);
                  mlir::Value section = fir::getBase(exV);
                  auto mem = builder.create<fir::AllocaOp>(
                      operandLocation, section.getType(), /*pinned=*/false);
                  builder.create<fir::StoreOp>(operandLocation, section, mem);
                  operands.push_back(mem);
                } else if (Fortran::parser::Unwrap<
                               Fortran::parser::StructureComponent>(
                               designator)) {
                  // Derived type components.
                  fir::ExtendedValue fieldAddr =
                      converter.genExprAddr(operandLocation, *expr, stmtCtx);
                  operands.push_back(fir::getBase(fieldAddr));
                } else {
                  // Scalar or full array.
                  if (const auto *dataRef{std::get_if<Fortran::parser::DataRef>(
                          &designator.u)}) {
                    const Fortran::parser::Name &name =
                        Fortran::parser::GetLastName(*dataRef);
                    addOperands(*name.symbol);
                  } else { // Unsupported
                    TODO(operandLocation,
                         "Unsupported type of OpenACC operand");
                  }
                }
              }
            },
            [&](const Fortran::parser::Name &name) {
              addOperands(*name.symbol);
            }},
        accObject.u);
  }
}

template <typename Clause>
static void genObjectListWithModifier(
    const Clause *x, Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semanticsContext,
    Fortran::lower::StatementContext &stmtCtx,
    Fortran::parser::AccDataModifier::Modifier mod,
    llvm::SmallVectorImpl<mlir::Value> &operandsWithModifier,
    llvm::SmallVectorImpl<mlir::Value> &operands) {
  const Fortran::parser::AccObjectListWithModifier &listWithModifier = x->v;
  const auto &accObjectList =
      std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
  const auto &modifier =
      std::get<std::optional<Fortran::parser::AccDataModifier>>(
          listWithModifier.t);
  if (modifier && (*modifier).v == mod) {
    genObjectList(accObjectList, converter, semanticsContext, stmtCtx,
                  operandsWithModifier);
  } else {
    genObjectList(accObjectList, converter, semanticsContext, stmtCtx,
                  operands);
  }
}

static void
addOperands(llvm::SmallVectorImpl<mlir::Value> &operands,
            llvm::SmallVectorImpl<int32_t> &operandSegments,
            const llvm::SmallVectorImpl<mlir::Value> &clauseOperands) {
  operands.append(clauseOperands.begin(), clauseOperands.end());
  operandSegments.push_back(clauseOperands.size());
}

static void addOperand(llvm::SmallVectorImpl<mlir::Value> &operands,
                       llvm::SmallVectorImpl<int32_t> &operandSegments,
                       const mlir::Value &clauseOperand) {
  if (clauseOperand) {
    operands.push_back(clauseOperand);
    operandSegments.push_back(1);
  } else {
    operandSegments.push_back(0);
  }
}

template <typename Op, typename Terminator>
static Op
createRegionOp(fir::FirOpBuilder &builder, mlir::Location loc,
               const llvm::SmallVectorImpl<mlir::Value> &operands,
               const llvm::SmallVectorImpl<int32_t> &operandSegments) {
  llvm::ArrayRef<mlir::Type> argTy;
  Op op = builder.create<Op>(loc, argTy, operands);
  builder.createBlock(&op.getRegion());
  mlir::Block &block = op.getRegion().back();
  builder.setInsertionPointToStart(&block);
  builder.create<Terminator>(loc);

  op->setAttr(Op::getOperandSegmentSizeAttr(),
              builder.getDenseI32ArrayAttr(operandSegments));

  // Place the insertion point to the start of the first block.
  builder.setInsertionPointToStart(&block);

  return op;
}

template <typename Op>
static Op
createSimpleOp(fir::FirOpBuilder &builder, mlir::Location loc,
               const llvm::SmallVectorImpl<mlir::Value> &operands,
               const llvm::SmallVectorImpl<int32_t> &operandSegments) {
  llvm::ArrayRef<mlir::Type> argTy;
  Op op = builder.create<Op>(loc, argTy, operands);
  op->setAttr(Op::getOperandSegmentSizeAttr(),
              builder.getDenseI32ArrayAttr(operandSegments));
  return op;
}

static void genAsyncClause(Fortran::lower::AbstractConverter &converter,
                           const Fortran::parser::AccClause::Async *asyncClause,
                           mlir::Value &async, bool &addAsyncAttr,
                           Fortran::lower::StatementContext &stmtCtx) {
  const auto &asyncClauseValue = asyncClause->v;
  if (asyncClauseValue) { // async has a value.
    async = fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(*asyncClauseValue), stmtCtx));
  } else {
    addAsyncAttr = true;
  }
}

static void genDeviceTypeClause(
    Fortran::lower::AbstractConverter &converter, mlir::Location clauseLocation,
    const Fortran::parser::AccClause::DeviceType *deviceTypeClause,
    llvm::SmallVectorImpl<mlir::Value> &operands,
    Fortran::lower::StatementContext &stmtCtx) {
  const Fortran::parser::AccDeviceTypeExprList &deviceTypeExprList =
      deviceTypeClause->v;
  for (const auto &deviceTypeExpr : deviceTypeExprList.v) {
    const auto &expr = std::get<std::optional<Fortran::parser::ScalarIntExpr>>(
        deviceTypeExpr.t);
    if (expr) {
      operands.push_back(fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(expr), stmtCtx, &clauseLocation)));
    } else {
      // * was passed as value and will be represented as a special constant.
      fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
      mlir::Value star = firOpBuilder.createIntegerConstant(
          clauseLocation, firOpBuilder.getIndexType(), starCst);
      operands.push_back(star);
    }
  }
}

static void genIfClause(Fortran::lower::AbstractConverter &converter,
                        mlir::Location clauseLocation,
                        const Fortran::parser::AccClause::If *ifClause,
                        mlir::Value &ifCond,
                        Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Value cond = fir::getBase(converter.genExprValue(
      *Fortran::semantics::GetExpr(ifClause->v), stmtCtx, &clauseLocation));
  ifCond = firOpBuilder.createConvert(clauseLocation, firOpBuilder.getI1Type(),
                                      cond);
}

static void genWaitClause(Fortran::lower::AbstractConverter &converter,
                          const Fortran::parser::AccClause::Wait *waitClause,
                          llvm::SmallVectorImpl<mlir::Value> &operands,
                          mlir::Value &waitDevnum, bool &addWaitAttr,
                          Fortran::lower::StatementContext &stmtCtx) {
  const auto &waitClauseValue = waitClause->v;
  if (waitClauseValue) { // wait has a value.
    const Fortran::parser::AccWaitArgument &waitArg = *waitClauseValue;
    const auto &waitList =
        std::get<std::list<Fortran::parser::ScalarIntExpr>>(waitArg.t);
    for (const Fortran::parser::ScalarIntExpr &value : waitList) {
      mlir::Value v = fir::getBase(
          converter.genExprValue(*Fortran::semantics::GetExpr(value), stmtCtx));
      operands.push_back(v);
    }

    const auto &waitDevnumValue =
        std::get<std::optional<Fortran::parser::ScalarIntExpr>>(waitArg.t);
    if (waitDevnumValue)
      waitDevnum = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(*waitDevnumValue), stmtCtx));
  } else {
    addWaitAttr = true;
  }
}

static mlir::acc::LoopOp
createLoopOp(Fortran::lower::AbstractConverter &converter,
             mlir::Location currentLocation,
             Fortran::semantics::SemanticsContext &semanticsContext,
             Fortran::lower::StatementContext &stmtCtx,
             const Fortran::parser::AccClauseList &accClauseList) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  mlir::Value workerNum;
  mlir::Value vectorNum;
  mlir::Value gangNum;
  mlir::Value gangStatic;
  llvm::SmallVector<mlir::Value, 2> tileOperands, privateOperands,
      reductionOperands;
  bool hasGang = false, hasVector = false, hasWorker = false;

  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *gangClause =
            std::get_if<Fortran::parser::AccClause::Gang>(&clause.u)) {
      if (gangClause->v) {
        const Fortran::parser::AccGangArgument &x = *gangClause->v;
        if (const auto &gangNumValue =
                std::get<std::optional<Fortran::parser::ScalarIntExpr>>(x.t)) {
          gangNum = fir::getBase(converter.genExprValue(
              *Fortran::semantics::GetExpr(gangNumValue.value()), stmtCtx));
        }
        if (const auto &gangStaticValue =
                std::get<std::optional<Fortran::parser::AccSizeExpr>>(x.t)) {
          const auto &expr =
              std::get<std::optional<Fortran::parser::ScalarIntExpr>>(
                  gangStaticValue.value().t);
          if (expr) {
            gangStatic = fir::getBase(converter.genExprValue(
                *Fortran::semantics::GetExpr(*expr), stmtCtx));
          } else {
            // * was passed as value and will be represented as a special
            // constant.
            gangStatic = firOpBuilder.createIntegerConstant(
                clauseLocation, firOpBuilder.getIndexType(), starCst);
          }
        }
      }
      hasGang = true;
    } else if (const auto *workerClause =
                   std::get_if<Fortran::parser::AccClause::Worker>(&clause.u)) {
      if (workerClause->v) {
        workerNum = fir::getBase(converter.genExprValue(
            *Fortran::semantics::GetExpr(*workerClause->v), stmtCtx));
      }
      hasWorker = true;
    } else if (const auto *vectorClause =
                   std::get_if<Fortran::parser::AccClause::Vector>(&clause.u)) {
      if (vectorClause->v) {
        vectorNum = fir::getBase(converter.genExprValue(
            *Fortran::semantics::GetExpr(*vectorClause->v), stmtCtx));
      }
      hasVector = true;
    } else if (const auto *tileClause =
                   std::get_if<Fortran::parser::AccClause::Tile>(&clause.u)) {
      const Fortran::parser::AccTileExprList &accTileExprList = tileClause->v;
      for (const auto &accTileExpr : accTileExprList.v) {
        const auto &expr =
            std::get<std::optional<Fortran::parser::ScalarIntConstantExpr>>(
                accTileExpr.t);
        if (expr) {
          tileOperands.push_back(fir::getBase(converter.genExprValue(
              *Fortran::semantics::GetExpr(*expr), stmtCtx)));
        } else {
          // * was passed as value and will be represented as a -1 constant
          // integer.
          mlir::Value tileStar = firOpBuilder.createIntegerConstant(
              clauseLocation, firOpBuilder.getIntegerType(32),
              /* STAR */ -1);
          tileOperands.push_back(tileStar);
        }
      }
    } else if (const auto *privateClause =
                   std::get_if<Fortran::parser::AccClause::Private>(
                       &clause.u)) {
      genObjectList(privateClause->v, converter, semanticsContext, stmtCtx,
                    privateOperands);
    }
    // Reduction clause is left out for the moment as the clause will probably
    // end up having its own operation.
  }

  // Prepare the operand segement size attribute and the operands value range.
  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<int32_t> operandSegments;
  addOperand(operands, operandSegments, gangNum);
  addOperand(operands, operandSegments, gangStatic);
  addOperand(operands, operandSegments, workerNum);
  addOperand(operands, operandSegments, vectorNum);
  addOperands(operands, operandSegments, tileOperands);
  addOperands(operands, operandSegments, privateOperands);
  addOperands(operands, operandSegments, reductionOperands);

  auto loopOp = createRegionOp<mlir::acc::LoopOp, mlir::acc::YieldOp>(
      firOpBuilder, currentLocation, operands, operandSegments);

  if (hasGang)
    loopOp.setHasGangAttr(firOpBuilder.getUnitAttr());
  if (hasWorker)
    loopOp.setHasWorkerAttr(firOpBuilder.getUnitAttr());
  if (hasVector)
    loopOp.setHasVectorAttr(firOpBuilder.getUnitAttr());

  // Lower clauses mapped to attributes
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    if (const auto *collapseClause =
            std::get_if<Fortran::parser::AccClause::Collapse>(&clause.u)) {
      const auto *expr = Fortran::semantics::GetExpr(collapseClause->v);
      const std::optional<int64_t> collapseValue =
          Fortran::evaluate::ToInt64(*expr);
      if (collapseValue) {
        loopOp.setCollapseAttr(firOpBuilder.getI64IntegerAttr(*collapseValue));
      }
    } else if (std::get_if<Fortran::parser::AccClause::Seq>(&clause.u)) {
      loopOp.setSeqAttr(firOpBuilder.getUnitAttr());
    } else if (std::get_if<Fortran::parser::AccClause::Independent>(
                   &clause.u)) {
      loopOp.setIndependentAttr(firOpBuilder.getUnitAttr());
    } else if (std::get_if<Fortran::parser::AccClause::Auto>(&clause.u)) {
      loopOp->setAttr(mlir::acc::LoopOp::getAutoAttrStrName(),
                      firOpBuilder.getUnitAttr());
    }
  }
  return loopOp;
}

static void genACC(Fortran::lower::AbstractConverter &converter,
                   Fortran::semantics::SemanticsContext &semanticsContext,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenACCLoopConstruct &loopConstruct) {

  const auto &beginLoopDirective =
      std::get<Fortran::parser::AccBeginLoopDirective>(loopConstruct.t);
  const auto &loopDirective =
      std::get<Fortran::parser::AccLoopDirective>(beginLoopDirective.t);

  mlir::Location currentLocation =
      converter.genLocation(beginLoopDirective.source);
  Fortran::lower::StatementContext stmtCtx;

  if (loopDirective.v == llvm::acc::ACCD_loop) {
    const auto &accClauseList =
        std::get<Fortran::parser::AccClauseList>(beginLoopDirective.t);
    createLoopOp(converter, currentLocation, semanticsContext, stmtCtx,
                 accClauseList);
  }
}

template <typename Op>
static Op
createComputeOp(Fortran::lower::AbstractConverter &converter,
                mlir::Location currentLocation,
                Fortran::semantics::SemanticsContext &semanticsContext,
                Fortran::lower::StatementContext &stmtCtx,
                const Fortran::parser::AccClauseList &accClauseList) {

  // Parallel operation operands
  mlir::Value async;
  mlir::Value numGangs;
  mlir::Value numWorkers;
  mlir::Value vectorLength;
  mlir::Value ifCond;
  mlir::Value selfCond;
  mlir::Value waitDevnum;
  llvm::SmallVector<mlir::Value, 2> waitOperands, reductionOperands,
      copyOperands, copyinOperands, copyinReadonlyOperands, copyoutOperands,
      copyoutZeroOperands, createOperands, createZeroOperands, noCreateOperands,
      presentOperands, devicePtrOperands, attachOperands, firstprivateOperands,
      privateOperands, dataClauseOperands;

  // Async, wait and self clause have optional values but can be present with
  // no value as well. When there is no value, the op has an attribute to
  // represent the clause.
  bool addAsyncAttr = false;
  bool addWaitAttr = false;
  bool addSelfAttr = false;

  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  // Lower clauses values mapped to operands.
  // Keep track of each group of operands separatly as clauses can appear
  // more than once.
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *asyncClause =
            std::get_if<Fortran::parser::AccClause::Async>(&clause.u)) {
      genAsyncClause(converter, asyncClause, async, addAsyncAttr, stmtCtx);
    } else if (const auto *waitClause =
                   std::get_if<Fortran::parser::AccClause::Wait>(&clause.u)) {
      genWaitClause(converter, waitClause, waitOperands, waitDevnum,
                    addWaitAttr, stmtCtx);
    } else if (const auto *numGangsClause =
                   std::get_if<Fortran::parser::AccClause::NumGangs>(
                       &clause.u)) {
      numGangs = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(numGangsClause->v), stmtCtx));
    } else if (const auto *numWorkersClause =
                   std::get_if<Fortran::parser::AccClause::NumWorkers>(
                       &clause.u)) {
      numWorkers = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(numWorkersClause->v), stmtCtx));
    } else if (const auto *vectorLengthClause =
                   std::get_if<Fortran::parser::AccClause::VectorLength>(
                       &clause.u)) {
      vectorLength = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(vectorLengthClause->v), stmtCtx));
    } else if (const auto *ifClause =
                   std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      genIfClause(converter, clauseLocation, ifClause, ifCond, stmtCtx);
    } else if (const auto *selfClause =
                   std::get_if<Fortran::parser::AccClause::Self>(&clause.u)) {
      const std::optional<Fortran::parser::AccSelfClause> &accSelfClause =
          selfClause->v;
      if (accSelfClause) {
        if (const auto *optCondition =
                std::get_if<std::optional<Fortran::parser::ScalarLogicalExpr>>(
                    &(*accSelfClause).u)) {
          if (*optCondition) {
            mlir::Value cond = fir::getBase(converter.genExprValue(
                *Fortran::semantics::GetExpr(*optCondition), stmtCtx));
            selfCond = firOpBuilder.createConvert(
                clauseLocation, firOpBuilder.getI1Type(), cond);
          }
        } else if (const auto *accClauseList =
                       std::get_if<Fortran::parser::AccObjectList>(
                           &(*accSelfClause).u)) {
          // TODO This would be nicer to be done in canonicalization step.
          if (accClauseList->v.size() == 1) {
            const auto &accObject = accClauseList->v.front();
            if (const auto *designator =
                    std::get_if<Fortran::parser::Designator>(&accObject.u)) {
              if (const auto *name = getDesignatorNameIfDataRef(*designator)) {
                auto cond = converter.getSymbolAddress(*name->symbol);
                selfCond = firOpBuilder.createConvert(
                    clauseLocation, firOpBuilder.getI1Type(), cond);
              }
            }
          }
        }
      } else {
        addSelfAttr = true;
      }
    } else if (const auto *copyClause =
                   std::get_if<Fortran::parser::AccClause::Copy>(&clause.u)) {
      genObjectList(copyClause->v, converter, semanticsContext, stmtCtx,
                    copyOperands);
    } else if (const auto *copyinClause =
                   std::get_if<Fortran::parser::AccClause::Copyin>(&clause.u)) {
      genObjectListWithModifier<Fortran::parser::AccClause::Copyin>(
          copyinClause, converter, semanticsContext, stmtCtx,
          Fortran::parser::AccDataModifier::Modifier::ReadOnly,
          copyinReadonlyOperands, copyinOperands);
    } else if (const auto *copyoutClause =
                   std::get_if<Fortran::parser::AccClause::Copyout>(
                       &clause.u)) {
      genObjectListWithModifier<Fortran::parser::AccClause::Copyout>(
          copyoutClause, converter, semanticsContext, stmtCtx,
          Fortran::parser::AccDataModifier::Modifier::Zero, copyoutZeroOperands,
          copyoutOperands);
    } else if (const auto *createClause =
                   std::get_if<Fortran::parser::AccClause::Create>(&clause.u)) {
      genObjectListWithModifier<Fortran::parser::AccClause::Create>(
          createClause, converter, semanticsContext, stmtCtx,
          Fortran::parser::AccDataModifier::Modifier::Zero, createZeroOperands,
          createOperands);
    } else if (const auto *noCreateClause =
                   std::get_if<Fortran::parser::AccClause::NoCreate>(
                       &clause.u)) {
      genObjectList(noCreateClause->v, converter, semanticsContext, stmtCtx,
                    noCreateOperands);
    } else if (const auto *presentClause =
                   std::get_if<Fortran::parser::AccClause::Present>(
                       &clause.u)) {
      genObjectList(presentClause->v, converter, semanticsContext, stmtCtx,
                    presentOperands);
    } else if (const auto *devicePtrClause =
                   std::get_if<Fortran::parser::AccClause::Deviceptr>(
                       &clause.u)) {
      genObjectList(devicePtrClause->v, converter, semanticsContext, stmtCtx,
                    devicePtrOperands);
    } else if (const auto *attachClause =
                   std::get_if<Fortran::parser::AccClause::Attach>(&clause.u)) {
      genObjectList(attachClause->v, converter, semanticsContext, stmtCtx,
                    attachOperands);
    } else if (const auto *privateClause =
                   std::get_if<Fortran::parser::AccClause::Private>(
                       &clause.u)) {
      genObjectList(privateClause->v, converter, semanticsContext, stmtCtx,
                    privateOperands);
    } else if (const auto *firstprivateClause =
                   std::get_if<Fortran::parser::AccClause::Firstprivate>(
                       &clause.u)) {
      genObjectList(firstprivateClause->v, converter, semanticsContext, stmtCtx,
                    firstprivateOperands);
    }
  }

  // Prepare the operand segement size attribute and the operands value range.
  llvm::SmallVector<mlir::Value, 8> operands;
  llvm::SmallVector<int32_t, 8> operandSegments;
  addOperand(operands, operandSegments, async);
  addOperands(operands, operandSegments, waitOperands);
  if constexpr (!std::is_same_v<Op, mlir::acc::SerialOp>) {
    addOperand(operands, operandSegments, numGangs);
    addOperand(operands, operandSegments, numWorkers);
    addOperand(operands, operandSegments, vectorLength);
  }
  addOperand(operands, operandSegments, ifCond);
  addOperand(operands, operandSegments, selfCond);
  if constexpr (!std::is_same_v<Op, mlir::acc::KernelsOp>)
    addOperands(operands, operandSegments, reductionOperands);
  addOperands(operands, operandSegments, copyOperands);
  addOperands(operands, operandSegments, copyinOperands);
  addOperands(operands, operandSegments, copyinReadonlyOperands);
  addOperands(operands, operandSegments, copyoutOperands);
  addOperands(operands, operandSegments, copyoutZeroOperands);
  addOperands(operands, operandSegments, createOperands);
  addOperands(operands, operandSegments, createZeroOperands);
  addOperands(operands, operandSegments, noCreateOperands);
  addOperands(operands, operandSegments, presentOperands);
  addOperands(operands, operandSegments, devicePtrOperands);
  addOperands(operands, operandSegments, attachOperands);
  if constexpr (!std::is_same_v<Op, mlir::acc::KernelsOp>) {
    addOperands(operands, operandSegments, privateOperands);
    addOperands(operands, operandSegments, firstprivateOperands);
  }
  addOperands(operands, operandSegments, dataClauseOperands);

  Op computeOp;
  if constexpr (std::is_same_v<Op, mlir::acc::KernelsOp>)
    computeOp = createRegionOp<Op, mlir::acc::TerminatorOp>(
        firOpBuilder, currentLocation, operands, operandSegments);
  else
    computeOp = createRegionOp<Op, mlir::acc::YieldOp>(
        firOpBuilder, currentLocation, operands, operandSegments);

  if (addAsyncAttr)
    computeOp.setAsyncAttrAttr(firOpBuilder.getUnitAttr());
  if (addWaitAttr)
    computeOp.setWaitAttrAttr(firOpBuilder.getUnitAttr());
  if (addSelfAttr)
    computeOp.setSelfAttrAttr(firOpBuilder.getUnitAttr());

  return computeOp;
}

static void genACCDataOp(Fortran::lower::AbstractConverter &converter,
                         mlir::Location currentLocation,
                         Fortran::semantics::SemanticsContext &semanticsContext,
                         Fortran::lower::StatementContext &stmtCtx,
                         const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value ifCond;
  llvm::SmallVector<mlir::Value> copyOperands, copyinOperands,
      copyinReadonlyOperands, copyoutOperands, copyoutZeroOperands,
      createOperands, createZeroOperands, noCreateOperands, presentOperands,
      deviceptrOperands, attachOperands, dataClauseOperands;

  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  // Lower clauses values mapped to operands.
  // Keep track of each group of operands separatly as clauses can appear
  // more than once.
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      genIfClause(converter, clauseLocation, ifClause, ifCond, stmtCtx);
    } else if (const auto *copyClause =
                   std::get_if<Fortran::parser::AccClause::Copy>(&clause.u)) {
      genObjectList(copyClause->v, converter, semanticsContext, stmtCtx,
                    copyOperands);
    } else if (const auto *copyinClause =
                   std::get_if<Fortran::parser::AccClause::Copyin>(&clause.u)) {
      genObjectListWithModifier<Fortran::parser::AccClause::Copyin>(
          copyinClause, converter, semanticsContext, stmtCtx,
          Fortran::parser::AccDataModifier::Modifier::ReadOnly,
          copyinReadonlyOperands, copyinOperands);
    } else if (const auto *copyoutClause =
                   std::get_if<Fortran::parser::AccClause::Copyout>(
                       &clause.u)) {
      genObjectListWithModifier<Fortran::parser::AccClause::Copyout>(
          copyoutClause, converter, semanticsContext, stmtCtx,
          Fortran::parser::AccDataModifier::Modifier::Zero, copyoutZeroOperands,
          copyoutOperands);
    } else if (const auto *createClause =
                   std::get_if<Fortran::parser::AccClause::Create>(&clause.u)) {
      genObjectListWithModifier<Fortran::parser::AccClause::Create>(
          createClause, converter, semanticsContext, stmtCtx,
          Fortran::parser::AccDataModifier::Modifier::Zero, createZeroOperands,
          createOperands);
    } else if (const auto *noCreateClause =
                   std::get_if<Fortran::parser::AccClause::NoCreate>(
                       &clause.u)) {
      genObjectList(noCreateClause->v, converter, semanticsContext, stmtCtx,
                    noCreateOperands);
    } else if (const auto *presentClause =
                   std::get_if<Fortran::parser::AccClause::Present>(
                       &clause.u)) {
      genObjectList(presentClause->v, converter, semanticsContext, stmtCtx,
                    presentOperands);
    } else if (const auto *deviceptrClause =
                   std::get_if<Fortran::parser::AccClause::Deviceptr>(
                       &clause.u)) {
      genObjectList(deviceptrClause->v, converter, semanticsContext, stmtCtx,
                    deviceptrOperands);
    } else if (const auto *attachClause =
                   std::get_if<Fortran::parser::AccClause::Attach>(&clause.u)) {
      genObjectList(attachClause->v, converter, semanticsContext, stmtCtx,
                    attachOperands);
    }
  }

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<int32_t> operandSegments;
  addOperand(operands, operandSegments, ifCond);
  addOperands(operands, operandSegments, copyOperands);
  addOperands(operands, operandSegments, copyinOperands);
  addOperands(operands, operandSegments, copyinReadonlyOperands);
  addOperands(operands, operandSegments, copyoutOperands);
  addOperands(operands, operandSegments, copyoutZeroOperands);
  addOperands(operands, operandSegments, createOperands);
  addOperands(operands, operandSegments, createZeroOperands);
  addOperands(operands, operandSegments, noCreateOperands);
  addOperands(operands, operandSegments, presentOperands);
  addOperands(operands, operandSegments, deviceptrOperands);
  addOperands(operands, operandSegments, attachOperands);
  addOperands(operands, operandSegments, dataClauseOperands);

  createRegionOp<mlir::acc::DataOp, mlir::acc::TerminatorOp>(
      firOpBuilder, currentLocation, operands, operandSegments);
}

static void
genACC(Fortran::lower::AbstractConverter &converter,
       Fortran::semantics::SemanticsContext &semanticsContext,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenACCBlockConstruct &blockConstruct) {
  const auto &beginBlockDirective =
      std::get<Fortran::parser::AccBeginBlockDirective>(blockConstruct.t);
  const auto &blockDirective =
      std::get<Fortran::parser::AccBlockDirective>(beginBlockDirective.t);
  const auto &accClauseList =
      std::get<Fortran::parser::AccClauseList>(beginBlockDirective.t);

  mlir::Location currentLocation = converter.genLocation(blockDirective.source);
  Fortran::lower::StatementContext stmtCtx;

  if (blockDirective.v == llvm::acc::ACCD_parallel) {
    createComputeOp<mlir::acc::ParallelOp>(
        converter, currentLocation, semanticsContext, stmtCtx, accClauseList);
  } else if (blockDirective.v == llvm::acc::ACCD_data) {
    genACCDataOp(converter, currentLocation, semanticsContext, stmtCtx,
                 accClauseList);
  } else if (blockDirective.v == llvm::acc::ACCD_serial) {
    createComputeOp<mlir::acc::SerialOp>(
        converter, currentLocation, semanticsContext, stmtCtx, accClauseList);
  } else if (blockDirective.v == llvm::acc::ACCD_kernels) {
    createComputeOp<mlir::acc::KernelsOp>(
        converter, currentLocation, semanticsContext, stmtCtx, accClauseList);
  } else if (blockDirective.v == llvm::acc::ACCD_host_data) {
    TODO(currentLocation, "host_data construct lowering");
  }
}

static void
genACC(Fortran::lower::AbstractConverter &converter,
       Fortran::semantics::SemanticsContext &semanticsContext,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenACCCombinedConstruct &combinedConstruct) {
  const auto &beginCombinedDirective =
      std::get<Fortran::parser::AccBeginCombinedDirective>(combinedConstruct.t);
  const auto &combinedDirective =
      std::get<Fortran::parser::AccCombinedDirective>(beginCombinedDirective.t);
  const auto &accClauseList =
      std::get<Fortran::parser::AccClauseList>(beginCombinedDirective.t);

  mlir::Location currentLocation =
      converter.genLocation(beginCombinedDirective.source);
  Fortran::lower::StatementContext stmtCtx;

  if (combinedDirective.v == llvm::acc::ACCD_kernels_loop) {
    createComputeOp<mlir::acc::KernelsOp>(
        converter, currentLocation, semanticsContext, stmtCtx, accClauseList);
    createLoopOp(converter, currentLocation, semanticsContext, stmtCtx,
                 accClauseList);
  } else if (combinedDirective.v == llvm::acc::ACCD_parallel_loop) {
    createComputeOp<mlir::acc::ParallelOp>(
        converter, currentLocation, semanticsContext, stmtCtx, accClauseList);
    createLoopOp(converter, currentLocation, semanticsContext, stmtCtx,
                 accClauseList);
  } else if (combinedDirective.v == llvm::acc::ACCD_serial_loop) {
    createComputeOp<mlir::acc::SerialOp>(
        converter, currentLocation, semanticsContext, stmtCtx, accClauseList);
    createLoopOp(converter, currentLocation, semanticsContext, stmtCtx,
                 accClauseList);
  } else {
    llvm::report_fatal_error("Unknown combined construct encountered");
  }
}

static void
genACCEnterDataOp(Fortran::lower::AbstractConverter &converter,
                  mlir::Location currentLocation,
                  Fortran::semantics::SemanticsContext &semanticsContext,
                  Fortran::lower::StatementContext &stmtCtx,
                  const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value ifCond, async, waitDevnum;
  llvm::SmallVector<mlir::Value> copyinOperands, createOperands,
      createZeroOperands, attachOperands, waitOperands, dataClauseOperands;

  // Async, wait and self clause have optional values but can be present with
  // no value as well. When there is no value, the op has an attribute to
  // represent the clause.
  bool addAsyncAttr = false;
  bool addWaitAttr = false;

  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  // Lower clauses values mapped to operands.
  // Keep track of each group of operands separatly as clauses can appear
  // more than once.
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      genIfClause(converter, clauseLocation, ifClause, ifCond, stmtCtx);
    } else if (const auto *asyncClause =
                   std::get_if<Fortran::parser::AccClause::Async>(&clause.u)) {
      genAsyncClause(converter, asyncClause, async, addAsyncAttr, stmtCtx);
    } else if (const auto *waitClause =
                   std::get_if<Fortran::parser::AccClause::Wait>(&clause.u)) {
      genWaitClause(converter, waitClause, waitOperands, waitDevnum,
                    addWaitAttr, stmtCtx);
    } else if (const auto *copyinClause =
                   std::get_if<Fortran::parser::AccClause::Copyin>(&clause.u)) {
      const Fortran::parser::AccObjectListWithModifier &listWithModifier =
          copyinClause->v;
      const auto &accObjectList =
          std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
      genObjectList(accObjectList, converter, semanticsContext, stmtCtx,
                    copyinOperands);
    } else if (const auto *createClause =
                   std::get_if<Fortran::parser::AccClause::Create>(&clause.u)) {
      genObjectListWithModifier<Fortran::parser::AccClause::Create>(
          createClause, converter, semanticsContext, stmtCtx,
          Fortran::parser::AccDataModifier::Modifier::Zero, createZeroOperands,
          createOperands);
    } else if (const auto *attachClause =
                   std::get_if<Fortran::parser::AccClause::Attach>(&clause.u)) {
      genObjectList(attachClause->v, converter, semanticsContext, stmtCtx,
                    attachOperands);
    } else {
      llvm::report_fatal_error(
          "Unknown clause in ENTER DATA directive lowering");
    }
  }

  // Prepare the operand segement size attribute and the operands value range.
  llvm::SmallVector<mlir::Value, 16> operands;
  llvm::SmallVector<int32_t, 8> operandSegments;
  addOperand(operands, operandSegments, ifCond);
  addOperand(operands, operandSegments, async);
  addOperand(operands, operandSegments, waitDevnum);
  addOperands(operands, operandSegments, waitOperands);
  addOperands(operands, operandSegments, copyinOperands);
  addOperands(operands, operandSegments, createOperands);
  addOperands(operands, operandSegments, createZeroOperands);
  addOperands(operands, operandSegments, attachOperands);
  addOperands(operands, operandSegments, dataClauseOperands);

  mlir::acc::EnterDataOp enterDataOp = createSimpleOp<mlir::acc::EnterDataOp>(
      firOpBuilder, currentLocation, operands, operandSegments);

  if (addAsyncAttr)
    enterDataOp.setAsyncAttr(firOpBuilder.getUnitAttr());
  if (addWaitAttr)
    enterDataOp.setWaitAttr(firOpBuilder.getUnitAttr());
}

static void
genACCExitDataOp(Fortran::lower::AbstractConverter &converter,
                 mlir::Location currentLocation,
                 Fortran::semantics::SemanticsContext &semanticsContext,
                 Fortran::lower::StatementContext &stmtCtx,
                 const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value ifCond, async, waitDevnum;
  llvm::SmallVector<mlir::Value> copyoutOperands, deleteOperands,
      detachOperands, waitOperands, dataClauseOperands;

  // Async and wait clause have optional values but can be present with
  // no value as well. When there is no value, the op has an attribute to
  // represent the clause.
  bool addAsyncAttr = false;
  bool addWaitAttr = false;
  bool addFinalizeAttr = false;

  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  // Lower clauses values mapped to operands.
  // Keep track of each group of operands separatly as clauses can appear
  // more than once.
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      genIfClause(converter, clauseLocation, ifClause, ifCond, stmtCtx);
    } else if (const auto *asyncClause =
                   std::get_if<Fortran::parser::AccClause::Async>(&clause.u)) {
      genAsyncClause(converter, asyncClause, async, addAsyncAttr, stmtCtx);
    } else if (const auto *waitClause =
                   std::get_if<Fortran::parser::AccClause::Wait>(&clause.u)) {
      genWaitClause(converter, waitClause, waitOperands, waitDevnum,
                    addWaitAttr, stmtCtx);
    } else if (const auto *copyoutClause =
                   std::get_if<Fortran::parser::AccClause::Copyout>(
                       &clause.u)) {
      const Fortran::parser::AccObjectListWithModifier &listWithModifier =
          copyoutClause->v;
      const auto &accObjectList =
          std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
      genObjectList(accObjectList, converter, semanticsContext, stmtCtx,
                    copyoutOperands);
    } else if (const auto *deleteClause =
                   std::get_if<Fortran::parser::AccClause::Delete>(&clause.u)) {
      genObjectList(deleteClause->v, converter, semanticsContext, stmtCtx,
                    deleteOperands);
    } else if (const auto *detachClause =
                   std::get_if<Fortran::parser::AccClause::Detach>(&clause.u)) {
      genObjectList(detachClause->v, converter, semanticsContext, stmtCtx,
                    detachOperands);
    } else if (std::get_if<Fortran::parser::AccClause::Finalize>(&clause.u)) {
      addFinalizeAttr = true;
    }
  }

  // Prepare the operand segement size attribute and the operands value range.
  llvm::SmallVector<mlir::Value, 14> operands;
  llvm::SmallVector<int32_t, 7> operandSegments;
  addOperand(operands, operandSegments, ifCond);
  addOperand(operands, operandSegments, async);
  addOperand(operands, operandSegments, waitDevnum);
  addOperands(operands, operandSegments, waitOperands);
  addOperands(operands, operandSegments, copyoutOperands);
  addOperands(operands, operandSegments, deleteOperands);
  addOperands(operands, operandSegments, detachOperands);
  addOperands(operands, operandSegments, dataClauseOperands);

  mlir::acc::ExitDataOp exitDataOp = createSimpleOp<mlir::acc::ExitDataOp>(
      firOpBuilder, currentLocation, operands, operandSegments);

  if (addAsyncAttr)
    exitDataOp.setAsyncAttr(firOpBuilder.getUnitAttr());
  if (addWaitAttr)
    exitDataOp.setWaitAttr(firOpBuilder.getUnitAttr());
  if (addFinalizeAttr)
    exitDataOp.setFinalizeAttr(firOpBuilder.getUnitAttr());
}

template <typename Op>
static void
genACCInitShutdownOp(Fortran::lower::AbstractConverter &converter,
                     mlir::Location currentLocation,
                     const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value ifCond, deviceNum;
  llvm::SmallVector<mlir::Value> deviceTypeOperands;

  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;

  // Lower clauses values mapped to operands.
  // Keep track of each group of operands separatly as clauses can appear
  // more than once.
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      genIfClause(converter, clauseLocation, ifClause, ifCond, stmtCtx);
    } else if (const auto *deviceNumClause =
                   std::get_if<Fortran::parser::AccClause::DeviceNum>(
                       &clause.u)) {
      deviceNum = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(deviceNumClause->v), stmtCtx));
    } else if (const auto *deviceTypeClause =
                   std::get_if<Fortran::parser::AccClause::DeviceType>(
                       &clause.u)) {
      genDeviceTypeClause(converter, clauseLocation, deviceTypeClause,
                          deviceTypeOperands, stmtCtx);
    }
  }

  // Prepare the operand segement size attribute and the operands value range.
  llvm::SmallVector<mlir::Value, 6> operands;
  llvm::SmallVector<int32_t, 3> operandSegments;
  addOperands(operands, operandSegments, deviceTypeOperands);
  addOperand(operands, operandSegments, deviceNum);
  addOperand(operands, operandSegments, ifCond);

  createSimpleOp<Op>(firOpBuilder, currentLocation, operands, operandSegments);
}

static void
genACCUpdateOp(Fortran::lower::AbstractConverter &converter,
               mlir::Location currentLocation,
               Fortran::semantics::SemanticsContext &semanticsContext,
               Fortran::lower::StatementContext &stmtCtx,
               const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value ifCond, async, waitDevnum;
  llvm::SmallVector<mlir::Value> hostOperands, deviceOperands, waitOperands,
      deviceTypeOperands;

  // Async and wait clause have optional values but can be present with
  // no value as well. When there is no value, the op has an attribute to
  // represent the clause.
  bool addAsyncAttr = false;
  bool addWaitAttr = false;
  bool addIfPresentAttr = false;

  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  // Lower clauses values mapped to operands.
  // Keep track of each group of operands separatly as clauses can appear
  // more than once.
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      genIfClause(converter, clauseLocation, ifClause, ifCond, stmtCtx);
    } else if (const auto *asyncClause =
                   std::get_if<Fortran::parser::AccClause::Async>(&clause.u)) {
      genAsyncClause(converter, asyncClause, async, addAsyncAttr, stmtCtx);
    } else if (const auto *waitClause =
                   std::get_if<Fortran::parser::AccClause::Wait>(&clause.u)) {
      genWaitClause(converter, waitClause, waitOperands, waitDevnum,
                    addWaitAttr, stmtCtx);
    } else if (const auto *deviceTypeClause =
                   std::get_if<Fortran::parser::AccClause::DeviceType>(
                       &clause.u)) {
      genDeviceTypeClause(converter, clauseLocation, deviceTypeClause,
                          deviceTypeOperands, stmtCtx);
    } else if (const auto *hostClause =
                   std::get_if<Fortran::parser::AccClause::Host>(&clause.u)) {
      genObjectList(hostClause->v, converter, semanticsContext, stmtCtx,
                    hostOperands);
    } else if (const auto *deviceClause =
                   std::get_if<Fortran::parser::AccClause::Device>(&clause.u)) {
      genObjectList(deviceClause->v, converter, semanticsContext, stmtCtx,
                    deviceOperands);
    }
  }

  // Prepare the operand segement size attribute and the operands value range.
  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<int32_t> operandSegments;
  addOperand(operands, operandSegments, ifCond);
  addOperand(operands, operandSegments, async);
  addOperand(operands, operandSegments, waitDevnum);
  addOperands(operands, operandSegments, waitOperands);
  addOperands(operands, operandSegments, deviceTypeOperands);
  addOperands(operands, operandSegments, hostOperands);
  addOperands(operands, operandSegments, deviceOperands);

  mlir::acc::UpdateOp updateOp = createSimpleOp<mlir::acc::UpdateOp>(
      firOpBuilder, currentLocation, operands, operandSegments);

  if (addAsyncAttr)
    updateOp.setAsyncAttr(firOpBuilder.getUnitAttr());
  if (addWaitAttr)
    updateOp.setWaitAttr(firOpBuilder.getUnitAttr());
  if (addIfPresentAttr)
    updateOp.setIfPresentAttr(firOpBuilder.getUnitAttr());
}

static void
genACC(Fortran::lower::AbstractConverter &converter,
       Fortran::semantics::SemanticsContext &semanticsContext,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenACCStandaloneConstruct &standaloneConstruct) {
  const auto &standaloneDirective =
      std::get<Fortran::parser::AccStandaloneDirective>(standaloneConstruct.t);
  const auto &accClauseList =
      std::get<Fortran::parser::AccClauseList>(standaloneConstruct.t);

  mlir::Location currentLocation =
      converter.genLocation(standaloneDirective.source);
  Fortran::lower::StatementContext stmtCtx;

  if (standaloneDirective.v == llvm::acc::Directive::ACCD_enter_data) {
    genACCEnterDataOp(converter, currentLocation, semanticsContext, stmtCtx,
                      accClauseList);
  } else if (standaloneDirective.v == llvm::acc::Directive::ACCD_exit_data) {
    genACCExitDataOp(converter, currentLocation, semanticsContext, stmtCtx,
                     accClauseList);
  } else if (standaloneDirective.v == llvm::acc::Directive::ACCD_init) {
    genACCInitShutdownOp<mlir::acc::InitOp>(converter, currentLocation,
                                            accClauseList);
  } else if (standaloneDirective.v == llvm::acc::Directive::ACCD_shutdown) {
    genACCInitShutdownOp<mlir::acc::ShutdownOp>(converter, currentLocation,
                                                accClauseList);
  } else if (standaloneDirective.v == llvm::acc::Directive::ACCD_set) {
    TODO(currentLocation, "OpenACC set directive not lowered yet!");
  } else if (standaloneDirective.v == llvm::acc::Directive::ACCD_update) {
    genACCUpdateOp(converter, currentLocation, semanticsContext, stmtCtx,
                   accClauseList);
  }
}

static void genACC(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenACCWaitConstruct &waitConstruct) {

  const auto &waitArgument =
      std::get<std::optional<Fortran::parser::AccWaitArgument>>(
          waitConstruct.t);
  const auto &accClauseList =
      std::get<Fortran::parser::AccClauseList>(waitConstruct.t);

  mlir::Value ifCond, waitDevnum, async;
  llvm::SmallVector<mlir::Value> waitOperands;

  // Async clause have optional values but can be present with
  // no value as well. When there is no value, the op has an attribute to
  // represent the clause.
  bool addAsyncAttr = false;

  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.genLocation(waitConstruct.source);
  Fortran::lower::StatementContext stmtCtx;

  if (waitArgument) { // wait has a value.
    const Fortran::parser::AccWaitArgument &waitArg = *waitArgument;
    const auto &waitList =
        std::get<std::list<Fortran::parser::ScalarIntExpr>>(waitArg.t);
    for (const Fortran::parser::ScalarIntExpr &value : waitList) {
      mlir::Value v = fir::getBase(
          converter.genExprValue(*Fortran::semantics::GetExpr(value), stmtCtx));
      waitOperands.push_back(v);
    }

    const auto &waitDevnumValue =
        std::get<std::optional<Fortran::parser::ScalarIntExpr>>(waitArg.t);
    if (waitDevnumValue)
      waitDevnum = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(*waitDevnumValue), stmtCtx));
  }

  // Lower clauses values mapped to operands.
  // Keep track of each group of operands separatly as clauses can appear
  // more than once.
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      genIfClause(converter, clauseLocation, ifClause, ifCond, stmtCtx);
    } else if (const auto *asyncClause =
                   std::get_if<Fortran::parser::AccClause::Async>(&clause.u)) {
      genAsyncClause(converter, asyncClause, async, addAsyncAttr, stmtCtx);
    }
  }

  // Prepare the operand segement size attribute and the operands value range.
  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<int32_t> operandSegments;
  addOperands(operands, operandSegments, waitOperands);
  addOperand(operands, operandSegments, async);
  addOperand(operands, operandSegments, waitDevnum);
  addOperand(operands, operandSegments, ifCond);

  mlir::acc::WaitOp waitOp = createSimpleOp<mlir::acc::WaitOp>(
      firOpBuilder, currentLocation, operands, operandSegments);

  if (addAsyncAttr)
    waitOp.setAsyncAttr(firOpBuilder.getUnitAttr());
}

void Fortran::lower::genOpenACCConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semanticsContext,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenACCConstruct &accConstruct) {

  std::visit(
      common::visitors{
          [&](const Fortran::parser::OpenACCBlockConstruct &blockConstruct) {
            genACC(converter, semanticsContext, eval, blockConstruct);
          },
          [&](const Fortran::parser::OpenACCCombinedConstruct
                  &combinedConstruct) {
            genACC(converter, semanticsContext, eval, combinedConstruct);
          },
          [&](const Fortran::parser::OpenACCLoopConstruct &loopConstruct) {
            genACC(converter, semanticsContext, eval, loopConstruct);
          },
          [&](const Fortran::parser::OpenACCStandaloneConstruct
                  &standaloneConstruct) {
            genACC(converter, semanticsContext, eval, standaloneConstruct);
          },
          [&](const Fortran::parser::OpenACCCacheConstruct &cacheConstruct) {
            TODO(converter.genLocation(cacheConstruct.source),
                 "OpenACC Cache construct not lowered yet!");
          },
          [&](const Fortran::parser::OpenACCWaitConstruct &waitConstruct) {
            genACC(converter, eval, waitConstruct);
          },
          [&](const Fortran::parser::OpenACCAtomicConstruct &atomicConstruct) {
            TODO(converter.genLocation(atomicConstruct.source),
                 "OpenACC Atomic construct not lowered yet!");
          },
      },
      accConstruct.u);
}

void Fortran::lower::genOpenACCDeclarativeConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenACCDeclarativeConstruct &accDeclConstruct) {

  std::visit(
      common::visitors{
          [&](const Fortran::parser::OpenACCStandaloneDeclarativeConstruct
                  &standaloneDeclarativeConstruct) {
            TODO(converter.genLocation(standaloneDeclarativeConstruct.source),
                 "OpenACC Standalone Declarative construct not lowered yet!");
          },
          [&](const Fortran::parser::OpenACCRoutineConstruct
                  &routineConstruct) {
            TODO(converter.genLocation(routineConstruct.source),
                 "OpenACC Routine construct not lowered yet!");
          },
      },
      accDeclConstruct.u);
}
