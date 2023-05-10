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
#include "flang/Lower/Support/Utils.h"
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

/// Generate the acc.bounds operation from the descriptor information.
static llvm::SmallVector<mlir::Value>
genBoundsOpsFromBox(fir::FirOpBuilder &builder, mlir::Location loc,
                    Fortran::lower::AbstractConverter &converter,
                    fir::ExtendedValue dataExv, mlir::Value box) {
  llvm::SmallVector<mlir::Value> bounds;
  mlir::Type idxTy = builder.getIndexType();
  mlir::Type boundTy = builder.getType<mlir::acc::DataBoundsType>();
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  assert(box.getType().isa<fir::BaseBoxType>() && "expect firbox or fir.class");
  for (unsigned dim = 0; dim < dataExv.rank(); ++dim) {
    mlir::Value d = builder.createIntegerConstant(loc, idxTy, dim);
    mlir::Value baseLb =
        fir::factory::readLowerBound(builder, loc, dataExv, dim, one);
    auto dimInfo =
        builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, box, d);
    mlir::Value lb = builder.createIntegerConstant(loc, idxTy, 0);
    mlir::Value ub =
        builder.create<mlir::arith::SubIOp>(loc, dimInfo.getExtent(), one);
    mlir::Value bound = builder.create<mlir::acc::DataBoundsOp>(
        loc, boundTy, lb, ub, mlir::Value(), dimInfo.getByteStride(), true,
        baseLb);
    bounds.push_back(bound);
  }
  return bounds;
}

/// Generate acc.bounds operation for base array without any subscripts
/// provided.
static llvm::SmallVector<mlir::Value>
genBaseBoundsOps(fir::FirOpBuilder &builder, mlir::Location loc,
                 Fortran::lower::AbstractConverter &converter,
                 fir::ExtendedValue dataExv, mlir::Value baseAddr) {
  mlir::Type idxTy = builder.getIndexType();
  mlir::Type boundTy = builder.getType<mlir::acc::DataBoundsType>();
  llvm::SmallVector<mlir::Value> bounds;

  if (dataExv.rank() == 0)
    return bounds;

  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  for (std::size_t dim = 0; dim < dataExv.rank(); ++dim) {
    mlir::Value baseLb =
        fir::factory::readLowerBound(builder, loc, dataExv, dim, one);
    mlir::Value ext = fir::factory::readExtent(builder, loc, dataExv, dim);
    mlir::Value lb = builder.createIntegerConstant(loc, idxTy, 0);

    // ub = extent - 1
    mlir::Value ub = builder.create<mlir::arith::SubIOp>(loc, ext, one);
    mlir::Value bound = builder.create<mlir::acc::DataBoundsOp>(
        loc, boundTy, lb, ub, mlir::Value(), one, false, baseLb);
    bounds.push_back(bound);
  }
  return bounds;
}

/// Generate acc.bounds operations for an array section when subscripts are
/// provided.
static llvm::SmallVector<mlir::Value>
genBoundsOps(fir::FirOpBuilder &builder, mlir::Location loc,
             Fortran::lower::AbstractConverter &converter,
             Fortran::lower::StatementContext &stmtCtx,
             const std::list<Fortran::parser::SectionSubscript> &subscripts,
             std::stringstream &asFortran, fir::ExtendedValue &dataExv,
             mlir::Value baseAddr) {
  int dimension = 0;
  mlir::Type idxTy = builder.getIndexType();
  mlir::Type boundTy = builder.getType<mlir::acc::DataBoundsType>();
  llvm::SmallVector<mlir::Value> bounds;

  mlir::Value zero = builder.createIntegerConstant(loc, idxTy, 0);
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  for (const auto &subscript : subscripts) {
    if (const auto *triplet{
            std::get_if<Fortran::parser::SubscriptTriplet>(&subscript.u)}) {
      if (dimension != 0)
        asFortran << ',';
      mlir::Value lbound, ubound, extent;
      std::optional<std::int64_t> lval, uval;
      mlir::Value baseLb =
          fir::factory::readLowerBound(builder, loc, dataExv, dimension, one);
      bool defaultLb = baseLb == one;
      mlir::Value stride = one;
      bool strideInBytes = false;

      if (fir::unwrapRefType(baseAddr.getType()).isa<fir::BaseBoxType>()) {
        mlir::Value d = builder.createIntegerConstant(loc, idxTy, dimension);
        auto dimInfo = builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy,
                                                      baseAddr, d);
        stride = dimInfo.getByteStride();
        strideInBytes = true;
      }

      const auto &lower{std::get<0>(triplet->t)};
      if (lower) {
        lval = Fortran::semantics::GetIntValue(lower);
        if (lval) {
          if (defaultLb) {
            lbound = builder.createIntegerConstant(loc, idxTy, *lval - 1);
          } else {
            mlir::Value lb = builder.createIntegerConstant(loc, idxTy, *lval);
            lbound = builder.create<mlir::arith::SubIOp>(loc, lb, baseLb);
          }
          asFortran << *lval;
        } else {
          const Fortran::lower::SomeExpr *lexpr =
              Fortran::semantics::GetExpr(*lower);
          mlir::Value lb =
              fir::getBase(converter.genExprValue(loc, *lexpr, stmtCtx));
          lb = builder.createConvert(loc, baseLb.getType(), lb);
          lbound = builder.create<mlir::arith::SubIOp>(loc, lb, baseLb);
          asFortran << lexpr->AsFortran();
        }
      } else {
        lbound = defaultLb ? zero : baseLb;
      }
      asFortran << ':';
      const auto &upper{std::get<1>(triplet->t)};
      if (upper) {
        uval = Fortran::semantics::GetIntValue(upper);
        if (uval) {
          if (defaultLb) {
            ubound = builder.createIntegerConstant(loc, idxTy, *uval - 1);
          } else {
            mlir::Value ub = builder.createIntegerConstant(loc, idxTy, *uval);
            ubound = builder.create<mlir::arith::SubIOp>(loc, ub, baseLb);
          }
          asFortran << *uval;
        } else {
          const Fortran::lower::SomeExpr *uexpr =
              Fortran::semantics::GetExpr(*upper);
          mlir::Value ub =
              fir::getBase(converter.genExprValue(loc, *uexpr, stmtCtx));
          ub = builder.createConvert(loc, baseLb.getType(), ub);
          ubound = builder.create<mlir::arith::SubIOp>(loc, ub, baseLb);
          asFortran << uexpr->AsFortran();
        }
      }
      if (lower && upper) {
        if (lval && uval && *uval < *lval) {
          mlir::emitError(loc, "zero sized array section");
          break;
        } else if (std::get<2>(triplet->t)) {
          const auto &strideExpr{std::get<2>(triplet->t)};
          if (strideExpr) {
            mlir::emitError(loc, "stride cannot be specified on "
                                 "an OpenACC array section");
            break;
          }
        }
      }
      // ub = baseLb + extent - 1
      if (!ubound) {
        mlir::Value ext =
            fir::factory::readExtent(builder, loc, dataExv, dimension);
        mlir::Value lbExt =
            builder.create<mlir::arith::AddIOp>(loc, ext, baseLb);
        ubound = builder.create<mlir::arith::SubIOp>(loc, lbExt, one);
      }
      mlir::Value bound = builder.create<mlir::acc::DataBoundsOp>(
          loc, boundTy, lbound, ubound, extent, stride, strideInBytes, baseLb);
      bounds.push_back(bound);
      ++dimension;
    }
  }
  return bounds;
}

static mlir::Value
getDataOperandBaseAddr(Fortran::lower::AbstractConverter &converter,
                       fir::FirOpBuilder &builder,
                       Fortran::lower::SymbolRef sym, mlir::Location loc) {
  mlir::Value symAddr = converter.getSymbolAddress(sym);
  // TODO: Might need revisiting to handle for non-shared clauses
  if (!symAddr) {
    if (const auto *details =
            sym->detailsIf<Fortran::semantics::HostAssocDetails>())
      symAddr = converter.getSymbolAddress(details->symbol());
  }

  if (!symAddr)
    llvm::report_fatal_error("could not retrieve symbol address");

  if (auto boxTy =
          fir::unwrapRefType(symAddr.getType()).dyn_cast<fir::BaseBoxType>()) {
    if (boxTy.getEleTy().isa<fir::RecordType>())
      TODO(loc, "derived type");

    // Load the box when baseAddr is a `fir.ref<fir.box<T>>` or a
    // `fir.ref<fir.class<T>>` type.
    if (symAddr.getType().isa<fir::ReferenceType>())
      return builder.create<fir::LoadOp>(loc, symAddr);
  }
  return symAddr;
}

static mlir::Value gatherDataOperandAddrAndBounds(
    Fortran::lower::AbstractConverter &converter, fir::FirOpBuilder &builder,
    Fortran::semantics::SemanticsContext &semanticsContext,
    Fortran::lower::StatementContext &stmtCtx,
    const Fortran::parser::AccObject &accObject, mlir::Location operandLocation,
    std::stringstream &asFortran, llvm::SmallVector<mlir::Value> &bounds) {
  mlir::Value baseAddr;
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::Designator &designator) {
            if (auto expr{Fortran::semantics::AnalyzeExpr(semanticsContext,
                                                          designator)}) {
              if ((*expr).Rank() > 0 &&
                  Fortran::parser::Unwrap<Fortran::parser::ArrayElement>(
                      designator)) {
                const auto *arrayElement =
                    Fortran::parser::Unwrap<Fortran::parser::ArrayElement>(
                        designator);
                const auto *dataRef =
                    std::get_if<Fortran::parser::DataRef>(&designator.u);
                fir::ExtendedValue dataExv;
                if (Fortran::parser::Unwrap<
                        Fortran::parser::StructureComponent>(
                        arrayElement->base)) {
                  auto exprBase = Fortran::semantics::AnalyzeExpr(
                      semanticsContext, arrayElement->base);
                  dataExv = converter.genExprAddr(operandLocation, *exprBase,
                                                  stmtCtx);
                  baseAddr = fir::getBase(dataExv);
                  asFortran << (*exprBase).AsFortran();
                } else {
                  const Fortran::parser::Name &name =
                      Fortran::parser::GetLastName(*dataRef);
                  baseAddr = getDataOperandBaseAddr(
                      converter, builder, *name.symbol, operandLocation);
                  dataExv = converter.getSymbolExtendedValue(*name.symbol);
                  asFortran << name.ToString();
                }

                if (!arrayElement->subscripts.empty()) {
                  asFortran << '(';
                  bounds = genBoundsOps(builder, operandLocation, converter,
                                        stmtCtx, arrayElement->subscripts,
                                        asFortran, dataExv, baseAddr);
                }
                asFortran << ')';
              } else if (Fortran::parser::Unwrap<
                             Fortran::parser::StructureComponent>(designator)) {
                fir::ExtendedValue compExv =
                    converter.genExprAddr(operandLocation, *expr, stmtCtx);
                baseAddr = fir::getBase(compExv);
                if (fir::unwrapRefType(baseAddr.getType())
                        .isa<fir::SequenceType>())
                  bounds = genBaseBoundsOps(builder, operandLocation, converter,
                                            compExv, baseAddr);
                asFortran << (*expr).AsFortran();

                // If the component is an allocatable or pointer the result of
                // genExprAddr will be the result of a fir.box_addr operation.
                // Retrieve the box so we handle it like other descriptor.
                if (auto boxAddrOp = mlir::dyn_cast_or_null<fir::BoxAddrOp>(
                        baseAddr.getDefiningOp())) {
                  baseAddr = boxAddrOp.getVal();
                  bounds = genBoundsOpsFromBox(builder, operandLocation,
                                               converter, compExv, baseAddr);
                }
              } else {
                // Scalar or full array.
                if (const auto *dataRef{
                        std::get_if<Fortran::parser::DataRef>(&designator.u)}) {
                  const Fortran::parser::Name &name =
                      Fortran::parser::GetLastName(*dataRef);
                  fir::ExtendedValue dataExv =
                      converter.getSymbolExtendedValue(*name.symbol);
                  baseAddr = getDataOperandBaseAddr(
                      converter, builder, *name.symbol, operandLocation);
                  if (fir::unwrapRefType(baseAddr.getType())
                          .isa<fir::BaseBoxType>())
                    bounds = genBoundsOpsFromBox(builder, operandLocation,
                                                 converter, dataExv, baseAddr);
                  if (fir::unwrapRefType(baseAddr.getType())
                          .isa<fir::SequenceType>())
                    bounds = genBaseBoundsOps(builder, operandLocation,
                                              converter, dataExv, baseAddr);
                  asFortran << name.ToString();
                } else { // Unsupported
                  llvm::report_fatal_error(
                      "Unsupported type of OpenACC operand");
                }
              }
            }
          },
          [&](const Fortran::parser::Name &name) {
            baseAddr = getDataOperandBaseAddr(converter, builder, *name.symbol,
                                              operandLocation);
            asFortran << name.ToString();
          }},
      accObject.u);
  return baseAddr;
}

static mlir::Location
genOperandLocation(Fortran::lower::AbstractConverter &converter,
                   const Fortran::parser::AccObject &accObject) {
  mlir::Location loc = converter.genUnknownLocation();
  std::visit(Fortran::common::visitors{
                 [&](const Fortran::parser::Designator &designator) {
                   loc = converter.genLocation(designator.source);
                 },
                 [&](const Fortran::parser::Name &name) {
                   loc = converter.genLocation(name.source);
                 }},
             accObject.u);
  return loc;
}

template <typename Op>
static Op createDataEntryOp(fir::FirOpBuilder &builder, mlir::Location loc,
                            mlir::Value baseAddr, std::stringstream &name,
                            mlir::SmallVector<mlir::Value> bounds,
                            bool structured, mlir::acc::DataClause dataClause) {
  mlir::Value varPtrPtr;
  if (auto boxTy = baseAddr.getType().dyn_cast<fir::BaseBoxType>())
    baseAddr = builder.create<fir::BoxAddrOp>(loc, baseAddr);

  Op op = builder.create<Op>(loc, baseAddr.getType(), baseAddr);
  op.setNameAttr(builder.getStringAttr(name.str()));
  op.setStructured(structured);
  op.setDataClause(dataClause);

  unsigned insPos = 1;
  if (varPtrPtr)
    op->insertOperands(insPos++, varPtrPtr);
  if (bounds.size() > 0)
    op->insertOperands(insPos, bounds);
  op->setAttr(Op::getOperandSegmentSizeAttr(),
              builder.getDenseI32ArrayAttr(
                  {1, varPtrPtr ? 1 : 0, static_cast<int32_t>(bounds.size())}));
  return op;
}

template <typename Op>
static void
genDataOperandOperations(const Fortran::parser::AccObjectList &objectList,
                         Fortran::lower::AbstractConverter &converter,
                         Fortran::semantics::SemanticsContext &semanticsContext,
                         Fortran::lower::StatementContext &stmtCtx,
                         llvm::SmallVectorImpl<mlir::Value> &dataOperands,
                         mlir::acc::DataClause dataClause, bool structured) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  for (const auto &accObject : objectList.v) {
    llvm::SmallVector<mlir::Value> bounds;
    std::stringstream asFortran;
    mlir::Location operandLocation = genOperandLocation(converter, accObject);
    mlir::Value baseAddr = gatherDataOperandAddrAndBounds(
        converter, builder, semanticsContext, stmtCtx, accObject,
        operandLocation, asFortran, bounds);
    Op op = createDataEntryOp<Op>(builder, operandLocation, baseAddr, asFortran,
                                  bounds, structured, dataClause);
    dataOperands.push_back(op.getAccPtr());
  }
}

template <typename EntryOp, typename ExitOp>
static void genDataExitOperations(fir::FirOpBuilder &builder,
                                  llvm::SmallVector<mlir::Value> operands,
                                  bool structured, bool implicit) {
  for (mlir::Value operand : operands) {
    auto entryOp = mlir::dyn_cast_or_null<EntryOp>(operand.getDefiningOp());
    assert(entryOp && "data entry op expected");
    mlir::Value varPtr;
    if constexpr (std::is_same_v<ExitOp, mlir::acc::CopyoutOp> ||
                  std::is_same_v<ExitOp, mlir::acc::UpdateHostOp>)
      varPtr = entryOp.getVarPtr();
    builder.create<ExitOp>(entryOp.getLoc(), entryOp.getAccPtr(), varPtr,
                           entryOp.getBounds(), entryOp.getDataClause(),
                           structured, implicit,
                           builder.getStringAttr(*entryOp.getName()));
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
    } else if (std::get_if<Fortran::parser::AccClause::Reduction>(&clause.u)) {
      // Reduction clause is left out for the moment as the clause will probably
      // end up having its own operation.
      TODO(clauseLocation, "OpenACC compute construct reduction lowering");
    }
  }

  // Prepare the operand segment size attribute and the operands value range.
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

template <typename Op, typename Clause>
static void genDataOperandOperationsWithModifier(
    const Clause *x, Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semanticsContext,
    Fortran::lower::StatementContext &stmtCtx,
    Fortran::parser::AccDataModifier::Modifier mod,
    llvm::SmallVectorImpl<mlir::Value> &dataClauseOperands,
    const mlir::acc::DataClause clause,
    const mlir::acc::DataClause clauseWithModifier) {
  const Fortran::parser::AccObjectListWithModifier &listWithModifier = x->v;
  const auto &accObjectList =
      std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
  const auto &modifier =
      std::get<std::optional<Fortran::parser::AccDataModifier>>(
          listWithModifier.t);
  mlir::acc::DataClause dataClause =
      (modifier && (*modifier).v == mod) ? clauseWithModifier : clause;
  genDataOperandOperations<Op>(accObjectList, converter, semanticsContext,
                               stmtCtx, dataClauseOperands, dataClause,
                               /*structured=*/true);
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
  llvm::SmallVector<mlir::Value> waitOperands, attachEntryOperands,
      copyEntryOperands, copyoutEntryOperands, createEntryOperands,
      dataClauseOperands;

  // TODO: need to more work/design.
  llvm::SmallVector<mlir::Value> reductionOperands, privateOperands,
      firstprivateOperands;

  // Async, wait and self clause have optional values but can be present with
  // no value as well. When there is no value, the op has an attribute to
  // represent the clause.
  bool addAsyncAttr = false;
  bool addWaitAttr = false;
  bool addSelfAttr = false;

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

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
            selfCond = builder.createConvert(clauseLocation,
                                             builder.getI1Type(), cond);
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
                selfCond = builder.createConvert(clauseLocation,
                                                 builder.getI1Type(), cond);
              }
            }
          }
        }
      } else {
        addSelfAttr = true;
      }
    } else if (const auto *copyClause =
                   std::get_if<Fortran::parser::AccClause::Copy>(&clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperations<mlir::acc::CopyinOp>(
          copyClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_copy,
          /*structured=*/true);
      copyEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                               dataClauseOperands.end());
    } else if (const auto *copyinClause =
                   std::get_if<Fortran::parser::AccClause::Copyin>(&clause.u)) {
      genDataOperandOperationsWithModifier<mlir::acc::CopyinOp,
                                           Fortran::parser::AccClause::Copyin>(
          copyinClause, converter, semanticsContext, stmtCtx,
          Fortran::parser::AccDataModifier::Modifier::ReadOnly,
          dataClauseOperands, mlir::acc::DataClause::acc_copyin,
          mlir::acc::DataClause::acc_copyin_readonly);
    } else if (const auto *copyoutClause =
                   std::get_if<Fortran::parser::AccClause::Copyout>(
                       &clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperationsWithModifier<mlir::acc::CreateOp,
                                           Fortran::parser::AccClause::Copyout>(
          copyoutClause, converter, semanticsContext, stmtCtx,
          Fortran::parser::AccDataModifier::Modifier::ReadOnly,
          dataClauseOperands, mlir::acc::DataClause::acc_copyout,
          mlir::acc::DataClause::acc_copyout_zero);
      copyoutEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                  dataClauseOperands.end());
    } else if (const auto *createClause =
                   std::get_if<Fortran::parser::AccClause::Create>(&clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperationsWithModifier<mlir::acc::CreateOp,
                                           Fortran::parser::AccClause::Create>(
          createClause, converter, semanticsContext, stmtCtx,
          Fortran::parser::AccDataModifier::Modifier::Zero, dataClauseOperands,
          mlir::acc::DataClause::acc_create,
          mlir::acc::DataClause::acc_create_zero);
      createEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                 dataClauseOperands.end());
    } else if (const auto *noCreateClause =
                   std::get_if<Fortran::parser::AccClause::NoCreate>(
                       &clause.u)) {
      genDataOperandOperations<mlir::acc::NoCreateOp>(
          noCreateClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_no_create,
          /*structured=*/true);
    } else if (const auto *presentClause =
                   std::get_if<Fortran::parser::AccClause::Present>(
                       &clause.u)) {
      genDataOperandOperations<mlir::acc::PresentOp>(
          presentClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_present,
          /*structured=*/true);
    } else if (const auto *devicePtrClause =
                   std::get_if<Fortran::parser::AccClause::Deviceptr>(
                       &clause.u)) {
      genDataOperandOperations<mlir::acc::DevicePtrOp>(
          devicePtrClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_deviceptr,
          /*structured=*/true);
    } else if (const auto *attachClause =
                   std::get_if<Fortran::parser::AccClause::Attach>(&clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperations<mlir::acc::AttachOp>(
          attachClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_attach,
          /*structured=*/true);
      attachEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                 dataClauseOperands.end());
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
    } else if (std::get_if<Fortran::parser::AccClause::Reduction>(&clause.u)) {
      TODO(clauseLocation, "compute construct reduction clause lowering");
    }
  }

  // Prepare the operand segment size attribute and the operands value range.
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
  if constexpr (!std::is_same_v<Op, mlir::acc::KernelsOp>) {
    addOperands(operands, operandSegments, reductionOperands);
    addOperands(operands, operandSegments, privateOperands);
    addOperands(operands, operandSegments, firstprivateOperands);
  }
  addOperands(operands, operandSegments, dataClauseOperands);

  Op computeOp;
  if constexpr (std::is_same_v<Op, mlir::acc::KernelsOp>)
    computeOp = createRegionOp<Op, mlir::acc::TerminatorOp>(
        builder, currentLocation, operands, operandSegments);
  else
    computeOp = createRegionOp<Op, mlir::acc::YieldOp>(
        builder, currentLocation, operands, operandSegments);

  if (addAsyncAttr)
    computeOp.setAsyncAttrAttr(builder.getUnitAttr());
  if (addWaitAttr)
    computeOp.setWaitAttrAttr(builder.getUnitAttr());
  if (addSelfAttr)
    computeOp.setSelfAttrAttr(builder.getUnitAttr());

  auto insPt = builder.saveInsertionPoint();
  builder.setInsertionPointAfter(computeOp);

  // Create the exit operations after the region.
  genDataExitOperations<mlir::acc::CopyinOp, mlir::acc::CopyoutOp>(
      builder, copyEntryOperands, /*structured=*/true, /*implicit=*/false);
  genDataExitOperations<mlir::acc::CreateOp, mlir::acc::CopyoutOp>(
      builder, copyoutEntryOperands, /*structured=*/true, /*implicit=*/false);
  genDataExitOperations<mlir::acc::AttachOp, mlir::acc::DetachOp>(
      builder, attachEntryOperands, /*structured=*/true, /*implicit=*/false);
  genDataExitOperations<mlir::acc::CreateOp, mlir::acc::DeleteOp>(
      builder, createEntryOperands, /*structured=*/true, /*implicit=*/false);

  builder.restoreInsertionPoint(insPt);
  return computeOp;
}

static void genACCDataOp(Fortran::lower::AbstractConverter &converter,
                         mlir::Location currentLocation,
                         Fortran::semantics::SemanticsContext &semanticsContext,
                         Fortran::lower::StatementContext &stmtCtx,
                         const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value ifCond;
  llvm::SmallVector<mlir::Value> attachEntryOperands, createEntryOperands,
      copyEntryOperands, copyoutEntryOperands, dataClauseOperands;

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

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
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperations<mlir::acc::CopyinOp>(
          copyClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_copy,
          /*structured=*/true);
      copyEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                               dataClauseOperands.end());
    } else if (const auto *copyinClause =
                   std::get_if<Fortran::parser::AccClause::Copyin>(&clause.u)) {
      genDataOperandOperationsWithModifier<mlir::acc::CopyinOp,
                                           Fortran::parser::AccClause::Copyin>(
          copyinClause, converter, semanticsContext, stmtCtx,
          Fortran::parser::AccDataModifier::Modifier::ReadOnly,
          dataClauseOperands, mlir::acc::DataClause::acc_copyin,
          mlir::acc::DataClause::acc_copyin_readonly);
    } else if (const auto *copyoutClause =
                   std::get_if<Fortran::parser::AccClause::Copyout>(
                       &clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperationsWithModifier<mlir::acc::CreateOp,
                                           Fortran::parser::AccClause::Copyout>(
          copyoutClause, converter, semanticsContext, stmtCtx,
          Fortran::parser::AccDataModifier::Modifier::Zero, dataClauseOperands,
          mlir::acc::DataClause::acc_copyout,
          mlir::acc::DataClause::acc_copyout_zero);
      copyoutEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                  dataClauseOperands.end());
    } else if (const auto *createClause =
                   std::get_if<Fortran::parser::AccClause::Create>(&clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperationsWithModifier<mlir::acc::CreateOp,
                                           Fortran::parser::AccClause::Create>(
          createClause, converter, semanticsContext, stmtCtx,
          Fortran::parser::AccDataModifier::Modifier::Zero, dataClauseOperands,
          mlir::acc::DataClause::acc_create,
          mlir::acc::DataClause::acc_create_zero);
      createEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                 dataClauseOperands.end());
    } else if (const auto *noCreateClause =
                   std::get_if<Fortran::parser::AccClause::NoCreate>(
                       &clause.u)) {
      genDataOperandOperations<mlir::acc::NoCreateOp>(
          noCreateClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_no_create,
          /*structured=*/true);
    } else if (const auto *presentClause =
                   std::get_if<Fortran::parser::AccClause::Present>(
                       &clause.u)) {
      genDataOperandOperations<mlir::acc::PresentOp>(
          presentClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_present,
          /*structured=*/true);
    } else if (const auto *deviceptrClause =
                   std::get_if<Fortran::parser::AccClause::Deviceptr>(
                       &clause.u)) {
      genDataOperandOperations<mlir::acc::DevicePtrOp>(
          deviceptrClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_deviceptr,
          /*structured=*/true);
    } else if (const auto *attachClause =
                   std::get_if<Fortran::parser::AccClause::Attach>(&clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperations<mlir::acc::AttachOp>(
          attachClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_attach,
          /*structured=*/true);
      attachEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                 dataClauseOperands.end());
    }
  }

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<int32_t> operandSegments;
  addOperand(operands, operandSegments, ifCond);
  addOperands(operands, operandSegments, dataClauseOperands);

  auto dataOp = createRegionOp<mlir::acc::DataOp, mlir::acc::TerminatorOp>(
      builder, currentLocation, operands, operandSegments);

  auto insPt = builder.saveInsertionPoint();
  builder.setInsertionPointAfter(dataOp);

  // Create the exit operations after the region.
  genDataExitOperations<mlir::acc::CopyinOp, mlir::acc::CopyoutOp>(
      builder, copyEntryOperands, /*structured=*/true, /*implicit=*/false);
  genDataExitOperations<mlir::acc::CreateOp, mlir::acc::CopyoutOp>(
      builder, copyoutEntryOperands, /*structured=*/true, /*implicit=*/false);
  genDataExitOperations<mlir::acc::AttachOp, mlir::acc::DetachOp>(
      builder, attachEntryOperands, /*structured=*/true, /*implicit=*/false);
  genDataExitOperations<mlir::acc::CreateOp, mlir::acc::DeleteOp>(
      builder, createEntryOperands, /*structured=*/true, /*implicit=*/false);

  builder.restoreInsertionPoint(insPt);
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
  llvm::SmallVector<mlir::Value> waitOperands, dataClauseOperands;

  // Async, wait and self clause have optional values but can be present with
  // no value as well. When there is no value, the op has an attribute to
  // represent the clause.
  bool addAsyncAttr = false;
  bool addWaitAttr = false;

  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  // Lower clauses values mapped to operands.
  // Keep track of each group of operands separately as clauses can appear
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
      genDataOperandOperations<mlir::acc::CopyinOp>(
          accObjectList, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_copyin, false);
    } else if (const auto *createClause =
                   std::get_if<Fortran::parser::AccClause::Create>(&clause.u)) {
      const Fortran::parser::AccObjectListWithModifier &listWithModifier =
          createClause->v;
      const auto &accObjectList =
          std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
      const auto &modifier =
          std::get<std::optional<Fortran::parser::AccDataModifier>>(
              listWithModifier.t);
      mlir::acc::DataClause clause = mlir::acc::DataClause::acc_create;
      if (modifier &&
          (*modifier).v == Fortran::parser::AccDataModifier::Modifier::Zero)
        clause = mlir::acc::DataClause::acc_create_zero;
      genDataOperandOperations<mlir::acc::CreateOp>(
          accObjectList, converter, semanticsContext, stmtCtx,
          dataClauseOperands, clause, false);
    } else if (const auto *attachClause =
                   std::get_if<Fortran::parser::AccClause::Attach>(&clause.u)) {
      genDataOperandOperations<mlir::acc::AttachOp>(
          attachClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_attach, false);
    } else {
      llvm::report_fatal_error(
          "Unknown clause in ENTER DATA directive lowering");
    }
  }

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value, 16> operands;
  llvm::SmallVector<int32_t, 8> operandSegments;
  addOperand(operands, operandSegments, ifCond);
  addOperand(operands, operandSegments, async);
  addOperand(operands, operandSegments, waitDevnum);
  addOperands(operands, operandSegments, waitOperands);
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
  llvm::SmallVector<mlir::Value> waitOperands, dataClauseOperands,
      copyoutOperands, deleteOperands, detachOperands;

  // Async and wait clause have optional values but can be present with
  // no value as well. When there is no value, the op has an attribute to
  // represent the clause.
  bool addAsyncAttr = false;
  bool addWaitAttr = false;
  bool addFinalizeAttr = false;

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

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
      genDataOperandOperations<mlir::acc::GetDevicePtrOp>(
          accObjectList, converter, semanticsContext, stmtCtx, copyoutOperands,
          mlir::acc::DataClause::acc_copyout, false);
    } else if (const auto *deleteClause =
                   std::get_if<Fortran::parser::AccClause::Delete>(&clause.u)) {
      genDataOperandOperations<mlir::acc::GetDevicePtrOp>(
          deleteClause->v, converter, semanticsContext, stmtCtx, deleteOperands,
          mlir::acc::DataClause::acc_delete, false);
    } else if (const auto *detachClause =
                   std::get_if<Fortran::parser::AccClause::Detach>(&clause.u)) {
      genDataOperandOperations<mlir::acc::GetDevicePtrOp>(
          detachClause->v, converter, semanticsContext, stmtCtx, detachOperands,
          mlir::acc::DataClause::acc_detach, false);
    } else if (std::get_if<Fortran::parser::AccClause::Finalize>(&clause.u)) {
      addFinalizeAttr = true;
    }
  }

  dataClauseOperands.append(copyoutOperands);
  dataClauseOperands.append(deleteOperands);
  dataClauseOperands.append(detachOperands);

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value, 14> operands;
  llvm::SmallVector<int32_t, 7> operandSegments;
  addOperand(operands, operandSegments, ifCond);
  addOperand(operands, operandSegments, async);
  addOperand(operands, operandSegments, waitDevnum);
  addOperands(operands, operandSegments, waitOperands);
  addOperands(operands, operandSegments, dataClauseOperands);

  mlir::acc::ExitDataOp exitDataOp = createSimpleOp<mlir::acc::ExitDataOp>(
      builder, currentLocation, operands, operandSegments);

  if (addAsyncAttr)
    exitDataOp.setAsyncAttr(builder.getUnitAttr());
  if (addWaitAttr)
    exitDataOp.setWaitAttr(builder.getUnitAttr());
  if (addFinalizeAttr)
    exitDataOp.setFinalizeAttr(builder.getUnitAttr());

  genDataExitOperations<mlir::acc::GetDevicePtrOp, mlir::acc::CopyoutOp>(
      builder, copyoutOperands, /*structured=*/false, /*implicit=*/false);
  genDataExitOperations<mlir::acc::GetDevicePtrOp, mlir::acc::DeleteOp>(
      builder, deleteOperands, /*structured=*/false, /*implicit=*/false);
  genDataExitOperations<mlir::acc::GetDevicePtrOp, mlir::acc::DetachOp>(
      builder, detachOperands, /*structured=*/false, /*implicit=*/false);
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

  // Prepare the operand segment size attribute and the operands value range.
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
  llvm::SmallVector<mlir::Value> dataClauseOperands, updateHostOperands,
      waitOperands, deviceTypeOperands;

  // Async and wait clause have optional values but can be present with
  // no value as well. When there is no value, the op has an attribute to
  // represent the clause.
  bool addAsyncAttr = false;
  bool addWaitAttr = false;
  bool addIfPresentAttr = false;

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

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
      genDataOperandOperations<mlir::acc::GetDevicePtrOp>(
          hostClause->v, converter, semanticsContext, stmtCtx,
          updateHostOperands, mlir::acc::DataClause::acc_update_host, false);
    } else if (const auto *deviceClause =
                   std::get_if<Fortran::parser::AccClause::Device>(&clause.u)) {
      genDataOperandOperations<mlir::acc::UpdateDeviceOp>(
          deviceClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_update_device, false);
    } else if (std::get_if<Fortran::parser::AccClause::IfPresent>(&clause.u)) {
      addIfPresentAttr = true;
    } else if (const auto *selfClause =
                   std::get_if<Fortran::parser::AccClause::Self>(&clause.u)) {
      const std::optional<Fortran::parser::AccSelfClause> &accSelfClause =
          selfClause->v;
      const auto *accObjectList =
          std::get_if<Fortran::parser::AccObjectList>(&(*accSelfClause).u);
      assert(accObjectList && "expect AccObjectList");
      genDataOperandOperations<mlir::acc::GetDevicePtrOp>(
          *accObjectList, converter, semanticsContext, stmtCtx,
          updateHostOperands, mlir::acc::DataClause::acc_update_self, false);
    }
  }

  dataClauseOperands.append(updateHostOperands);

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<int32_t> operandSegments;
  addOperand(operands, operandSegments, ifCond);
  addOperand(operands, operandSegments, async);
  addOperand(operands, operandSegments, waitDevnum);
  addOperands(operands, operandSegments, waitOperands);
  addOperands(operands, operandSegments, deviceTypeOperands);
  addOperands(operands, operandSegments, dataClauseOperands);

  mlir::acc::UpdateOp updateOp = createSimpleOp<mlir::acc::UpdateOp>(
      builder, currentLocation, operands, operandSegments);

  genDataExitOperations<mlir::acc::GetDevicePtrOp, mlir::acc::UpdateHostOp>(
      builder, updateHostOperands, /*structured=*/false, /*implicit=*/false);

  if (addAsyncAttr)
    updateOp.setAsyncAttr(builder.getUnitAttr());
  if (addWaitAttr)
    updateOp.setWaitAttr(builder.getUnitAttr());
  if (addIfPresentAttr)
    updateOp.setIfPresentAttr(builder.getUnitAttr());
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

  // Prepare the operand segment size attribute and the operands value range.
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
