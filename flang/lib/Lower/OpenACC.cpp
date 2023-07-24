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
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/IntrinsicCall.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/tools.h"
#include "llvm/Frontend/OpenACC/ACC.h.inc"

// Special value for * passed in device_type or gang clauses.
static constexpr std::int64_t starCst = -1;

/// Generate the acc.bounds operation from the descriptor information.
static llvm::SmallVector<mlir::Value>
genBoundsOpsFromBox(fir::FirOpBuilder &builder, mlir::Location loc,
                    Fortran::lower::AbstractConverter &converter,
                    fir::ExtendedValue dataExv, mlir::Value box) {
  llvm::SmallVector<mlir::Value> bounds;
  mlir::Type idxTy = builder.getIndexType();
  mlir::Type boundTy = builder.getType<mlir::acc::DataBoundsType>();
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  assert(box.getType().isa<fir::BaseBoxType>() &&
         "expect fir.box or fir.class");
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
        loc, boundTy, lb, ub, ext, one, false, baseLb);
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
                            bool structured, mlir::acc::DataClause dataClause,
                            mlir::Type retTy) {
  mlir::Value varPtrPtr;
  if (auto boxTy = baseAddr.getType().dyn_cast<fir::BaseBoxType>()) {
    baseAddr = builder.create<fir::BoxAddrOp>(loc, baseAddr);
    retTy = baseAddr.getType();
  }

  Op op = builder.create<Op>(loc, retTy, baseAddr);
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
                                  bounds, structured, dataClause,
                                  baseAddr.getType());
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

template <typename RecipeOp>
static void genPrivateLikeInitRegion(mlir::OpBuilder &builder, RecipeOp recipe,
                                     mlir::Type ty, mlir::Location loc) {
  mlir::Value retVal = recipe.getInitRegion().front().getArgument(0);
  if (auto refTy = mlir::dyn_cast_or_null<fir::ReferenceType>(ty)) {
    if (fir::isa_trivial(refTy.getEleTy()))
      retVal = builder.create<fir::AllocaOp>(loc, refTy.getEleTy());
    else if (auto seqTy =
                 mlir::dyn_cast_or_null<fir::SequenceType>(refTy.getEleTy())) {
      if (seqTy.hasDynamicExtents())
        TODO(loc, "private recipe of array with dynamic extents");
      if (fir::isa_trivial(seqTy.getEleTy()))
        retVal = builder.create<fir::AllocaOp>(loc, seqTy);
    }
  }
  builder.create<mlir::acc::YieldOp>(loc, retVal);
}

mlir::acc::PrivateRecipeOp
Fortran::lower::createOrGetPrivateRecipe(mlir::OpBuilder &builder,
                                         llvm::StringRef recipeName,
                                         mlir::Location loc, mlir::Type ty) {
  mlir::ModuleOp mod =
      builder.getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();
  if (auto recipe = mod.lookupSymbol<mlir::acc::PrivateRecipeOp>(recipeName))
    return recipe;

  auto crtPos = builder.saveInsertionPoint();
  mlir::OpBuilder modBuilder(mod.getBodyRegion());
  auto recipe =
      modBuilder.create<mlir::acc::PrivateRecipeOp>(loc, recipeName, ty);
  builder.createBlock(&recipe.getInitRegion(), recipe.getInitRegion().end(),
                      {ty}, {loc});
  builder.setInsertionPointToEnd(&recipe.getInitRegion().back());
  genPrivateLikeInitRegion<mlir::acc::PrivateRecipeOp>(builder, recipe, ty,
                                                       loc);
  builder.restoreInsertionPoint(crtPos);
  return recipe;
}

mlir::acc::FirstprivateRecipeOp Fortran::lower::createOrGetFirstprivateRecipe(
    mlir::OpBuilder &builder, llvm::StringRef recipeName, mlir::Location loc,
    mlir::Type ty) {
  mlir::ModuleOp mod =
      builder.getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();
  if (auto recipe =
          mod.lookupSymbol<mlir::acc::FirstprivateRecipeOp>(recipeName))
    return recipe;

  auto crtPos = builder.saveInsertionPoint();
  mlir::OpBuilder modBuilder(mod.getBodyRegion());
  auto recipe =
      modBuilder.create<mlir::acc::FirstprivateRecipeOp>(loc, recipeName, ty);
  builder.createBlock(&recipe.getInitRegion(), recipe.getInitRegion().end(),
                      {ty}, {loc});
  builder.setInsertionPointToEnd(&recipe.getInitRegion().back());
  genPrivateLikeInitRegion<mlir::acc::FirstprivateRecipeOp>(builder, recipe, ty,
                                                            loc);

  // Add empty copy region for firstprivate. TODO add copy sequence.
  builder.createBlock(&recipe.getCopyRegion(), recipe.getCopyRegion().end(),
                      {ty, ty}, {loc, loc});

  builder.setInsertionPointToEnd(&recipe.getCopyRegion().back());
  if (auto refTy = mlir::dyn_cast_or_null<fir::ReferenceType>(ty)) {
    if (fir::isa_trivial(refTy.getEleTy())) {
      mlir::Value initValue = builder.create<fir::LoadOp>(
          loc, recipe.getCopyRegion().front().getArgument(0));
      builder.create<fir::StoreOp>(
          loc, initValue, recipe.getCopyRegion().front().getArgument(1));
    } else if (auto seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(
                   refTy.getEleTy())) {
      if (seqTy.hasDynamicExtents())
        TODO(loc, "private recipe of array with dynamic extents");
      mlir::Type idxTy = builder.getIndexType();
      mlir::Type refTy = fir::ReferenceType::get(seqTy.getEleTy());
      mlir::Value arraySrc = recipe.getCopyRegion().front().getArgument(0);
      mlir::Value arrayDst = recipe.getCopyRegion().front().getArgument(1);
      llvm::SmallVector<fir::DoLoopOp> loops;
      llvm::SmallVector<mlir::Value> ivs;
      for (auto ext : llvm::reverse(seqTy.getShape())) {
        auto lb = builder.create<mlir::arith::ConstantOp>(
            loc, idxTy, builder.getIntegerAttr(idxTy, 0));
        auto ub = builder.create<mlir::arith::ConstantOp>(
            loc, idxTy, builder.getIntegerAttr(idxTy, ext - 1));
        auto step = builder.create<mlir::arith::ConstantOp>(
            loc, idxTy, builder.getIntegerAttr(idxTy, 1));
        auto loop = builder.create<fir::DoLoopOp>(loc, lb, ub, step,
                                                  /*unordered=*/false);
        builder.setInsertionPointToStart(loop.getBody());
        loops.push_back(loop);
        ivs.push_back(loop.getInductionVar());
      }
      auto addr1 = builder.create<fir::CoordinateOp>(loc, refTy, arraySrc, ivs);
      auto addr2 = builder.create<fir::CoordinateOp>(loc, refTy, arrayDst, ivs);
      auto loadedValue = builder.create<fir::LoadOp>(loc, addr1);
      builder.create<fir::StoreOp>(loc, loadedValue, addr2);
      builder.setInsertionPointAfter(loops[0]);
    }
  }
  builder.create<mlir::acc::TerminatorOp>(loc);
  builder.restoreInsertionPoint(crtPos);
  return recipe;
}

/// Rebuild the array type from the acc.bounds operation with constant
/// lowerbound/upperbound or extent.
mlir::Type getTypeFromBounds(llvm::SmallVector<mlir::Value> &bounds,
                             mlir::Type ty) {
  auto seqTy =
      mlir::dyn_cast_or_null<fir::SequenceType>(fir::unwrapRefType(ty));
  if (!bounds.empty() && seqTy) {
    llvm::SmallVector<int64_t> shape;
    for (auto b : bounds) {
      auto boundsOp =
          mlir::dyn_cast<mlir::acc::DataBoundsOp>(b.getDefiningOp());
      if (boundsOp.getLowerbound() &&
          fir::getIntIfConstant(boundsOp.getLowerbound()) &&
          boundsOp.getUpperbound() &&
          fir::getIntIfConstant(boundsOp.getUpperbound())) {
        int64_t ext = *fir::getIntIfConstant(boundsOp.getUpperbound()) -
                      *fir::getIntIfConstant(boundsOp.getLowerbound()) + 1;
        shape.push_back(ext);
      } else if (boundsOp.getExtent() &&
                 fir::getIntIfConstant(boundsOp.getExtent())) {
        shape.push_back(*fir::getIntIfConstant(boundsOp.getExtent()));
      } else {
        return ty; // TODO: handle dynamic shaped array slice.
      }
    }
    if (shape.empty() || shape.size() != bounds.size())
      return ty;
    auto newSeqTy = fir::SequenceType::get(shape, seqTy.getEleTy());
    if (mlir::isa<fir::ReferenceType>(ty))
      return fir::ReferenceType::get(newSeqTy);
    return newSeqTy;
  }
  return ty;
}

template <typename RecipeOp>
static void
genPrivatizations(const Fortran::parser::AccObjectList &objectList,
                  Fortran::lower::AbstractConverter &converter,
                  Fortran::semantics::SemanticsContext &semanticsContext,
                  Fortran::lower::StatementContext &stmtCtx,
                  llvm::SmallVectorImpl<mlir::Value> &dataOperands,
                  llvm::SmallVector<mlir::Attribute> &privatizations) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  for (const auto &accObject : objectList.v) {
    llvm::SmallVector<mlir::Value> bounds;
    std::stringstream asFortran;
    mlir::Location operandLocation = genOperandLocation(converter, accObject);
    mlir::Value baseAddr = gatherDataOperandAddrAndBounds(
        converter, builder, semanticsContext, stmtCtx, accObject,
        operandLocation, asFortran, bounds);

    RecipeOp recipe;
    mlir::Type retTy = getTypeFromBounds(bounds, baseAddr.getType());
    if constexpr (std::is_same_v<RecipeOp, mlir::acc::PrivateRecipeOp>) {
      std::string recipeName =
          fir::getTypeAsString(retTy, converter.getKindMap(), "privatization");
      recipe = Fortran::lower::createOrGetPrivateRecipe(builder, recipeName,
                                                        operandLocation, retTy);
      auto op = createDataEntryOp<mlir::acc::PrivateOp>(
          builder, operandLocation, baseAddr, asFortran, bounds, true,
          mlir::acc::DataClause::acc_private, retTy);
      dataOperands.push_back(op.getAccPtr());
    } else {
      std::string recipeName = fir::getTypeAsString(
          retTy, converter.getKindMap(), "firstprivatization");
      recipe = Fortran::lower::createOrGetFirstprivateRecipe(
          builder, recipeName, operandLocation, retTy);
      auto op = createDataEntryOp<mlir::acc::FirstprivateOp>(
          builder, operandLocation, baseAddr, asFortran, bounds, true,
          mlir::acc::DataClause::acc_firstprivate, retTy);
      dataOperands.push_back(op.getAccPtr());
    }
    privatizations.push_back(mlir::SymbolRefAttr::get(
        builder.getContext(), recipe.getSymName().str()));
  }
}

/// Return the corresponding enum value for the mlir::acc::ReductionOperator
/// from the parser representation.
static mlir::acc::ReductionOperator
getReductionOperator(const Fortran::parser::AccReductionOperator &op) {
  switch (op.v) {
  case Fortran::parser::AccReductionOperator::Operator::Plus:
    return mlir::acc::ReductionOperator::AccAdd;
  case Fortran::parser::AccReductionOperator::Operator::Multiply:
    return mlir::acc::ReductionOperator::AccMul;
  case Fortran::parser::AccReductionOperator::Operator::Max:
    return mlir::acc::ReductionOperator::AccMax;
  case Fortran::parser::AccReductionOperator::Operator::Min:
    return mlir::acc::ReductionOperator::AccMin;
  case Fortran::parser::AccReductionOperator::Operator::Iand:
    return mlir::acc::ReductionOperator::AccIand;
  case Fortran::parser::AccReductionOperator::Operator::Ior:
    return mlir::acc::ReductionOperator::AccIor;
  case Fortran::parser::AccReductionOperator::Operator::Ieor:
    return mlir::acc::ReductionOperator::AccXor;
  case Fortran::parser::AccReductionOperator::Operator::And:
    return mlir::acc::ReductionOperator::AccLand;
  case Fortran::parser::AccReductionOperator::Operator::Or:
    return mlir::acc::ReductionOperator::AccLor;
  case Fortran::parser::AccReductionOperator::Operator::Eqv:
    return mlir::acc::ReductionOperator::AccEqv;
  case Fortran::parser::AccReductionOperator::Operator::Neqv:
    return mlir::acc::ReductionOperator::AccNeqv;
  }
  llvm_unreachable("unexpected reduction operator");
}

/// Get the initial value for reduction operator.
template <typename R>
static R getReductionInitValue(mlir::acc::ReductionOperator op, mlir::Type ty) {
  if (op == mlir::acc::ReductionOperator::AccMin) {
    // min init value -> largest
    if constexpr (std::is_same_v<R, llvm::APInt>) {
      assert(ty.isIntOrIndex() && "expect integer or index type");
      return llvm::APInt::getSignedMaxValue(ty.getIntOrFloatBitWidth());
    }
    if constexpr (std::is_same_v<R, llvm::APFloat>) {
      auto floatTy = mlir::dyn_cast_or_null<mlir::FloatType>(ty);
      assert(floatTy && "expect float type");
      return llvm::APFloat::getLargest(floatTy.getFloatSemantics(),
                                       /*negative=*/false);
    }
  } else if (op == mlir::acc::ReductionOperator::AccMax) {
    // max init value -> smallest
    if constexpr (std::is_same_v<R, llvm::APInt>) {
      assert(ty.isIntOrIndex() && "expect integer or index type");
      return llvm::APInt::getSignedMinValue(ty.getIntOrFloatBitWidth());
    }
    if constexpr (std::is_same_v<R, llvm::APFloat>) {
      auto floatTy = mlir::dyn_cast_or_null<mlir::FloatType>(ty);
      assert(floatTy && "expect float type");
      return llvm::APFloat::getSmallest(floatTy.getFloatSemantics(),
                                        /*negative=*/true);
    }
  } else if (op == mlir::acc::ReductionOperator::AccIand) {
    if constexpr (std::is_same_v<R, llvm::APInt>) {
      assert(ty.isIntOrIndex() && "expect integer type");
      unsigned bits = ty.getIntOrFloatBitWidth();
      return llvm::APInt::getAllOnes(bits);
    }
  } else {
    // +, ior, ieor init value -> 0
    // * init value -> 1
    int64_t value = (op == mlir::acc::ReductionOperator::AccMul) ? 1 : 0;
    if constexpr (std::is_same_v<R, llvm::APInt>) {
      assert(ty.isIntOrIndex() && "expect integer or index type");
      return llvm::APInt(ty.getIntOrFloatBitWidth(), value, true);
    }

    if constexpr (std::is_same_v<R, llvm::APFloat>) {
      assert(mlir::isa<mlir::FloatType>(ty) && "expect float type");
      auto floatTy = mlir::dyn_cast<mlir::FloatType>(ty);
      return llvm::APFloat(floatTy.getFloatSemantics(), value);
    }

    if constexpr (std::is_same_v<R, int64_t>)
      return value;
  }
  llvm_unreachable("OpenACC reduction unsupported type");
}

/// Check if the DataBoundsOp is a constant bound (lb and ub are constants or
/// extent is a constant).
bool isConstantBound(mlir::acc::DataBoundsOp &op) {
  if (op.getLowerbound() && fir::getIntIfConstant(op.getLowerbound()) &&
      op.getUpperbound() && fir::getIntIfConstant(op.getUpperbound()))
    return true;
  if (op.getExtent() && fir::getIntIfConstant(op.getExtent()))
    return true;
  return false;
}

/// Determine if the bounds represent a dynamic shape.
bool hasDynamicShape(llvm::SmallVector<mlir::Value> &bounds) {
  if (bounds.empty())
    return false;
  for (auto b : bounds) {
    auto op = mlir::dyn_cast<mlir::acc::DataBoundsOp>(b.getDefiningOp());
    if (!isConstantBound(op))
      return true;
  }
  return false;
}

/// Return a constant with the initial value for the reduction operator and
/// type combination.
static mlir::Value getReductionInitValue(fir::FirOpBuilder &builder,
                                         mlir::Location loc, mlir::Type ty,
                                         mlir::acc::ReductionOperator op) {
  if (op == mlir::acc::ReductionOperator::AccLand ||
      op == mlir::acc::ReductionOperator::AccLor ||
      op == mlir::acc::ReductionOperator::AccEqv ||
      op == mlir::acc::ReductionOperator::AccNeqv) {
    assert(mlir::isa<fir::LogicalType>(ty) && "expect fir.logical type");
    bool value = true; // .true. for .and. and .eqv.
    if (op == mlir::acc::ReductionOperator::AccLor ||
        op == mlir::acc::ReductionOperator::AccNeqv)
      value = false; // .false. for .or. and .neqv.
    return builder.createBool(loc, value);
  }
  if (ty.isIntOrIndex())
    return builder.create<mlir::arith::ConstantOp>(
        loc, ty,
        builder.getIntegerAttr(ty, getReductionInitValue<llvm::APInt>(op, ty)));
  if (op == mlir::acc::ReductionOperator::AccMin ||
      op == mlir::acc::ReductionOperator::AccMax) {
    if (mlir::isa<fir::ComplexType>(ty))
      llvm::report_fatal_error(
          "min/max reduction not supported for complex type");
    if (auto floatTy = mlir::dyn_cast_or_null<mlir::FloatType>(ty))
      return builder.create<mlir::arith::ConstantOp>(
          loc, ty,
          builder.getFloatAttr(ty,
                               getReductionInitValue<llvm::APFloat>(op, ty)));
  } else if (auto floatTy = mlir::dyn_cast_or_null<mlir::FloatType>(ty)) {
    return builder.create<mlir::arith::ConstantOp>(
        loc, ty,
        builder.getFloatAttr(ty, getReductionInitValue<int64_t>(op, ty)));
  } else if (auto cmplxTy = mlir::dyn_cast_or_null<fir::ComplexType>(ty)) {
    mlir::Type floatTy =
        Fortran::lower::convertReal(builder.getContext(), cmplxTy.getFKind());
    mlir::Value realInit = builder.createRealConstant(
        loc, floatTy, getReductionInitValue<int64_t>(op, cmplxTy));
    mlir::Value imagInit = builder.createRealConstant(loc, floatTy, 0.0);
    return fir::factory::Complex{builder, loc}.createComplex(
        cmplxTy.getFKind(), realInit, imagInit);
  }

  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(ty))
    return getReductionInitValue(builder, loc, seqTy.getEleTy(), op);

  llvm::report_fatal_error("Unsupported OpenACC reduction type");
}

static mlir::Value genReductionInitRegion(fir::FirOpBuilder &builder,
                                          mlir::Location loc, mlir::Type ty,
                                          mlir::acc::ReductionOperator op) {
  ty = fir::unwrapRefType(ty);
  mlir::Value initValue = getReductionInitValue(builder, loc, ty, op);
  if (fir::isa_trivial(ty)) {
    mlir::Value alloca = builder.create<fir::AllocaOp>(loc, ty);
    builder.create<fir::StoreOp>(loc, builder.createConvert(loc, ty, initValue),
                                 alloca);
    return alloca;
  } else if (auto seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(ty)) {
    if (seqTy.hasDynamicExtents())
      TODO(loc, "private recipe of array with dynamic extents");
    if (fir::isa_trivial(seqTy.getEleTy())) {
      mlir::Value alloca = builder.create<fir::AllocaOp>(loc, seqTy);
      mlir::Type idxTy = builder.getIndexType();
      mlir::Type refTy = fir::ReferenceType::get(seqTy.getEleTy());
      llvm::SmallVector<fir::DoLoopOp> loops;
      llvm::SmallVector<mlir::Value> ivs;
      for (auto ext : llvm::reverse(seqTy.getShape())) {
        auto lb = builder.createIntegerConstant(loc, idxTy, 0);
        auto ub = builder.createIntegerConstant(loc, idxTy, ext - 1);
        auto step = builder.createIntegerConstant(loc, idxTy, 1);
        auto loop = builder.create<fir::DoLoopOp>(loc, lb, ub, step,
                                                  /*unordered=*/false);
        builder.setInsertionPointToStart(loop.getBody());
        loops.push_back(loop);
        ivs.push_back(loop.getInductionVar());
      }
      auto coord = builder.create<fir::CoordinateOp>(loc, refTy, alloca, ivs);
      builder.create<fir::StoreOp>(loc, initValue, coord);
      builder.setInsertionPointAfter(loops[0]);
      return alloca;
    }
  }
  llvm::report_fatal_error("Unsupported OpenACC reduction type");
}

template <typename Op>
static mlir::Value genLogicalCombiner(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Value value1,
                                      mlir::Value value2) {
  mlir::Type i1 = builder.getI1Type();
  mlir::Value v1 = builder.create<fir::ConvertOp>(loc, i1, value1);
  mlir::Value v2 = builder.create<fir::ConvertOp>(loc, i1, value2);
  mlir::Value combined = builder.create<Op>(loc, v1, v2);
  return builder.create<fir::ConvertOp>(loc, value1.getType(), combined);
}

static mlir::Value loadIfRef(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value value) {
  if (mlir::isa<fir::ReferenceType, fir::PointerType, fir::HeapType>(
          value.getType()))
    return builder.create<fir::LoadOp>(loc, value);
  return value;
}

static mlir::Value genComparisonCombiner(fir::FirOpBuilder &builder,
                                         mlir::Location loc,
                                         mlir::arith::CmpIPredicate pred,
                                         mlir::Value value1,
                                         mlir::Value value2) {
  mlir::Type i1 = builder.getI1Type();
  mlir::Value v1 = builder.create<fir::ConvertOp>(loc, i1, value1);
  mlir::Value v2 = builder.create<fir::ConvertOp>(loc, i1, value2);
  mlir::Value add = builder.create<mlir::arith::CmpIOp>(loc, pred, v1, v2);
  return builder.create<fir::ConvertOp>(loc, value1.getType(), add);
}

static mlir::Value genScalarCombiner(fir::FirOpBuilder &builder,
                                     mlir::Location loc,
                                     mlir::acc::ReductionOperator op,
                                     mlir::Type ty, mlir::Value value1,
                                     mlir::Value value2) {
  value1 = loadIfRef(builder, loc, value1);
  value2 = loadIfRef(builder, loc, value2);
  if (op == mlir::acc::ReductionOperator::AccAdd) {
    if (ty.isIntOrIndex())
      return builder.create<mlir::arith::AddIOp>(loc, value1, value2);
    if (mlir::isa<mlir::FloatType>(ty))
      return builder.create<mlir::arith::AddFOp>(loc, value1, value2);
    if (auto cmplxTy = mlir::dyn_cast_or_null<fir::ComplexType>(ty))
      return builder.create<fir::AddcOp>(loc, value1, value2);
    TODO(loc, "reduction add type");
  }

  if (op == mlir::acc::ReductionOperator::AccMul) {
    if (ty.isIntOrIndex())
      return builder.create<mlir::arith::MulIOp>(loc, value1, value2);
    if (mlir::isa<mlir::FloatType>(ty))
      return builder.create<mlir::arith::MulFOp>(loc, value1, value2);
    if (mlir::isa<fir::ComplexType>(ty))
      return builder.create<fir::MulcOp>(loc, value1, value2);
    TODO(loc, "reduction mul type");
  }

  if (op == mlir::acc::ReductionOperator::AccMin)
    return fir::genMin(builder, loc, {value1, value2});

  if (op == mlir::acc::ReductionOperator::AccMax)
    return fir::genMax(builder, loc, {value1, value2});

  if (op == mlir::acc::ReductionOperator::AccIand)
    return builder.create<mlir::arith::AndIOp>(loc, value1, value2);

  if (op == mlir::acc::ReductionOperator::AccIor)
    return builder.create<mlir::arith::OrIOp>(loc, value1, value2);

  if (op == mlir::acc::ReductionOperator::AccXor)
    return builder.create<mlir::arith::XOrIOp>(loc, value1, value2);

  if (op == mlir::acc::ReductionOperator::AccLand)
    return genLogicalCombiner<mlir::arith::AndIOp>(builder, loc, value1,
                                                   value2);

  if (op == mlir::acc::ReductionOperator::AccLor)
    return genLogicalCombiner<mlir::arith::OrIOp>(builder, loc, value1, value2);

  if (op == mlir::acc::ReductionOperator::AccEqv)
    return genComparisonCombiner(builder, loc, mlir::arith::CmpIPredicate::eq,
                                 value1, value2);

  if (op == mlir::acc::ReductionOperator::AccNeqv)
    return genComparisonCombiner(builder, loc, mlir::arith::CmpIPredicate::ne,
                                 value1, value2);

  TODO(loc, "reduction operator");
}

static void genCombiner(fir::FirOpBuilder &builder, mlir::Location loc,
                        mlir::acc::ReductionOperator op, mlir::Type ty,
                        mlir::Value value1, mlir::Value value2) {
  ty = fir::unwrapRefType(ty);

  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(ty)) {
    if (seqTy.hasDynamicExtents())
      TODO(loc, "OpenACC reduction on array with dynamic extents");
    mlir::Type idxTy = builder.getIndexType();
    mlir::Type refTy = fir::ReferenceType::get(seqTy.getEleTy());

    llvm::SmallVector<fir::DoLoopOp> loops;
    llvm::SmallVector<mlir::Value> ivs;
    for (auto ext : llvm::reverse(seqTy.getShape())) {
      auto lb = builder.createIntegerConstant(loc, idxTy, 0);
      auto ub = builder.createIntegerConstant(loc, idxTy, ext - 1);
      auto step = builder.createIntegerConstant(loc, idxTy, 1);
      auto loop = builder.create<fir::DoLoopOp>(loc, lb, ub, step,
                                                /*unordered=*/false);
      builder.setInsertionPointToStart(loop.getBody());
      loops.push_back(loop);
      ivs.push_back(loop.getInductionVar());
    }
    auto addr1 = builder.create<fir::CoordinateOp>(loc, refTy, value1, ivs);
    auto addr2 = builder.create<fir::CoordinateOp>(loc, refTy, value2, ivs);
    auto load1 = builder.create<fir::LoadOp>(loc, addr1);
    auto load2 = builder.create<fir::LoadOp>(loc, addr2);
    mlir::Value res =
        genScalarCombiner(builder, loc, op, seqTy.getEleTy(), load1, load2);
    builder.create<fir::StoreOp>(loc, res, addr1);
    builder.setInsertionPointAfter(loops[0]);
  } else {
    mlir::Value res = genScalarCombiner(builder, loc, op, ty, value1, value2);
    builder.create<fir::StoreOp>(loc, res, value1);
  }
}

mlir::acc::ReductionRecipeOp Fortran::lower::createOrGetReductionRecipe(
    fir::FirOpBuilder &builder, llvm::StringRef recipeName, mlir::Location loc,
    mlir::Type ty, mlir::acc::ReductionOperator op,
    llvm::SmallVector<mlir::Value> &bounds) {
  mlir::ModuleOp mod =
      builder.getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();
  if (auto recipe = mod.lookupSymbol<mlir::acc::ReductionRecipeOp>(recipeName))
    return recipe;

  auto crtPos = builder.saveInsertionPoint();
  mlir::OpBuilder modBuilder(mod.getBodyRegion());
  auto recipe =
      modBuilder.create<mlir::acc::ReductionRecipeOp>(loc, recipeName, ty, op);
  builder.createBlock(&recipe.getInitRegion(), recipe.getInitRegion().end(),
                      {ty}, {loc});
  builder.setInsertionPointToEnd(&recipe.getInitRegion().back());
  mlir::Value initValue = genReductionInitRegion(builder, loc, ty, op);
  builder.create<mlir::acc::YieldOp>(loc, initValue);

  builder.createBlock(&recipe.getCombinerRegion(),
                      recipe.getCombinerRegion().end(), {ty, ty}, {loc, loc});
  builder.setInsertionPointToEnd(&recipe.getCombinerRegion().back());
  mlir::Value v1 = recipe.getCombinerRegion().front().getArgument(0);
  mlir::Value v2 = recipe.getCombinerRegion().front().getArgument(1);
  genCombiner(builder, loc, op, ty, v1, v2);
  builder.create<mlir::acc::YieldOp>(loc, v1);
  builder.restoreInsertionPoint(crtPos);
  return recipe;
}

static void
genReductions(const Fortran::parser::AccObjectListWithReduction &objectList,
              Fortran::lower::AbstractConverter &converter,
              Fortran::semantics::SemanticsContext &semanticsContext,
              Fortran::lower::StatementContext &stmtCtx,
              llvm::SmallVectorImpl<mlir::Value> &reductionOperands,
              llvm::SmallVector<mlir::Attribute> &reductionRecipes) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  const auto &objects = std::get<Fortran::parser::AccObjectList>(objectList.t);
  const auto &op =
      std::get<Fortran::parser::AccReductionOperator>(objectList.t);
  mlir::acc::ReductionOperator mlirOp = getReductionOperator(op);
  for (const auto &accObject : objects.v) {
    llvm::SmallVector<mlir::Value> bounds;
    std::stringstream asFortran;
    mlir::Location operandLocation = genOperandLocation(converter, accObject);
    mlir::Value baseAddr = gatherDataOperandAddrAndBounds(
        converter, builder, semanticsContext, stmtCtx, accObject,
        operandLocation, asFortran, bounds);

    if (hasDynamicShape(bounds))
      TODO(operandLocation, "OpenACC reductions with dynamic shaped array");

    mlir::Type reductionTy = fir::unwrapRefType(baseAddr.getType());
    if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(reductionTy))
      reductionTy = seqTy.getEleTy();

    if (!fir::isa_trivial(reductionTy) &&
        ((fir::isAllocatableType(reductionTy) ||
          fir::isPointerType(reductionTy)) &&
         !bounds.empty()))
      TODO(operandLocation, "reduction with unsupported type");

    auto op = createDataEntryOp<mlir::acc::ReductionOp>(
        builder, operandLocation, baseAddr, asFortran, bounds,
        /*structured=*/true, mlir::acc::DataClause::acc_reduction,
        baseAddr.getType());
    mlir::Type ty = op.getAccPtr().getType();
    std::string recipeName = fir::getTypeAsString(
        ty, converter.getKindMap(),
        ("reduction_" + stringifyReductionOperator(mlirOp)).str());
    mlir::acc::ReductionRecipeOp recipe =
        Fortran::lower::createOrGetReductionRecipe(
            builder, recipeName, operandLocation, ty, mlirOp, bounds);
    reductionRecipes.push_back(mlir::SymbolRefAttr::get(
        builder.getContext(), recipe.getSymName().str()));
    reductionOperands.push_back(op.getAccPtr());
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
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  mlir::Value workerNum;
  mlir::Value vectorNum;
  mlir::Value gangNum;
  mlir::Value gangDim;
  mlir::Value gangStatic;
  llvm::SmallVector<mlir::Value, 2> tileOperands, privateOperands,
      reductionOperands;
  llvm::SmallVector<mlir::Attribute> privatizations, reductionRecipes;
  bool hasGang = false, hasVector = false, hasWorker = false;

  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *gangClause =
            std::get_if<Fortran::parser::AccClause::Gang>(&clause.u)) {
      if (gangClause->v) {
        const Fortran::parser::AccGangArgList &x = *gangClause->v;
        for (const Fortran::parser::AccGangArg &gangArg : x.v) {
          if (const auto *num =
                  std::get_if<Fortran::parser::AccGangArg::Num>(&gangArg.u)) {
            gangNum = fir::getBase(converter.genExprValue(
                *Fortran::semantics::GetExpr(num->v), stmtCtx));
          } else if (const auto *staticArg =
                         std::get_if<Fortran::parser::AccGangArg::Static>(
                             &gangArg.u)) {
            const Fortran::parser::AccSizeExpr &sizeExpr = staticArg->v;
            if (sizeExpr.v) {
              gangStatic = fir::getBase(converter.genExprValue(
                  *Fortran::semantics::GetExpr(*sizeExpr.v), stmtCtx));
            } else {
              // * was passed as value and will be represented as a special
              // constant.
              gangStatic = builder.createIntegerConstant(
                  clauseLocation, builder.getIndexType(), starCst);
            }
          } else if (const auto *dim =
                         std::get_if<Fortran::parser::AccGangArg::Dim>(
                             &gangArg.u)) {
            gangDim = fir::getBase(converter.genExprValue(
                *Fortran::semantics::GetExpr(dim->v), stmtCtx));
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
          mlir::Value tileStar = builder.createIntegerConstant(
              clauseLocation, builder.getIntegerType(32),
              /* STAR */ -1);
          tileOperands.push_back(tileStar);
        }
      }
    } else if (const auto *privateClause =
                   std::get_if<Fortran::parser::AccClause::Private>(
                       &clause.u)) {
      genPrivatizations<mlir::acc::PrivateRecipeOp>(
          privateClause->v, converter, semanticsContext, stmtCtx,
          privateOperands, privatizations);
    } else if (const auto *reductionClause =
                   std::get_if<Fortran::parser::AccClause::Reduction>(
                       &clause.u)) {
      genReductions(reductionClause->v, converter, semanticsContext, stmtCtx,
                    reductionOperands, reductionRecipes);
    }
  }

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<int32_t> operandSegments;
  addOperand(operands, operandSegments, gangNum);
  addOperand(operands, operandSegments, gangDim);
  addOperand(operands, operandSegments, gangStatic);
  addOperand(operands, operandSegments, workerNum);
  addOperand(operands, operandSegments, vectorNum);
  addOperands(operands, operandSegments, tileOperands);
  addOperands(operands, operandSegments, privateOperands);
  addOperands(operands, operandSegments, reductionOperands);

  auto loopOp = createRegionOp<mlir::acc::LoopOp, mlir::acc::YieldOp>(
      builder, currentLocation, operands, operandSegments);

  if (hasGang)
    loopOp.setHasGangAttr(builder.getUnitAttr());
  if (hasWorker)
    loopOp.setHasWorkerAttr(builder.getUnitAttr());
  if (hasVector)
    loopOp.setHasVectorAttr(builder.getUnitAttr());

  if (!privatizations.empty())
    loopOp.setPrivatizationsAttr(
        mlir::ArrayAttr::get(builder.getContext(), privatizations));

  if (!reductionRecipes.empty())
    loopOp.setReductionRecipesAttr(
        mlir::ArrayAttr::get(builder.getContext(), reductionRecipes));

  // Lower clauses mapped to attributes
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *collapseClause =
            std::get_if<Fortran::parser::AccClause::Collapse>(&clause.u)) {
      const Fortran::parser::AccCollapseArg &arg = collapseClause->v;
      const auto &force = std::get<bool>(arg.t);
      if (force)
        TODO(clauseLocation, "OpenACC collapse force modifier");
      const auto &intExpr =
          std::get<Fortran::parser::ScalarIntConstantExpr>(arg.t);
      const auto *expr = Fortran::semantics::GetExpr(intExpr);
      const std::optional<int64_t> collapseValue =
          Fortran::evaluate::ToInt64(*expr);
      if (collapseValue) {
        loopOp.setCollapseAttr(builder.getI64IntegerAttr(*collapseValue));
      }
    } else if (std::get_if<Fortran::parser::AccClause::Seq>(&clause.u)) {
      loopOp.setSeqAttr(builder.getUnitAttr());
    } else if (std::get_if<Fortran::parser::AccClause::Independent>(
                   &clause.u)) {
      loopOp.setIndependentAttr(builder.getUnitAttr());
    } else if (std::get_if<Fortran::parser::AccClause::Auto>(&clause.u)) {
      loopOp->setAttr(mlir::acc::LoopOp::getAutoAttrStrName(),
                      builder.getUnitAttr());
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
  mlir::Value numWorkers;
  mlir::Value vectorLength;
  mlir::Value ifCond;
  mlir::Value selfCond;
  mlir::Value waitDevnum;
  llvm::SmallVector<mlir::Value> waitOperands, attachEntryOperands,
      copyEntryOperands, copyoutEntryOperands, createEntryOperands,
      dataClauseOperands, numGangs;

  llvm::SmallVector<mlir::Value> reductionOperands, privateOperands,
      firstprivateOperands;
  llvm::SmallVector<mlir::Attribute> privatizations, firstPrivatizations,
      reductionRecipes;

  // Async, wait and self clause have optional values but can be present with
  // no value as well. When there is no value, the op has an attribute to
  // represent the clause.
  bool addAsyncAttr = false;
  bool addWaitAttr = false;
  bool addSelfAttr = false;

  bool hasDefaultNone = false;
  bool hasDefaultPresent = false;

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
      for (const Fortran::parser::ScalarIntExpr &expr : numGangsClause->v)
        numGangs.push_back(fir::getBase(converter.genExprValue(
            *Fortran::semantics::GetExpr(expr), stmtCtx)));
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
              if (const auto *name =
                      Fortran::semantics::getDesignatorNameIfDataRef(
                          *designator)) {
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
      genPrivatizations<mlir::acc::PrivateRecipeOp>(
          privateClause->v, converter, semanticsContext, stmtCtx,
          privateOperands, privatizations);
    } else if (const auto *firstprivateClause =
                   std::get_if<Fortran::parser::AccClause::Firstprivate>(
                       &clause.u)) {
      genPrivatizations<mlir::acc::FirstprivateRecipeOp>(
          firstprivateClause->v, converter, semanticsContext, stmtCtx,
          firstprivateOperands, firstPrivatizations);
    } else if (const auto *reductionClause =
                   std::get_if<Fortran::parser::AccClause::Reduction>(
                       &clause.u)) {
      genReductions(reductionClause->v, converter, semanticsContext, stmtCtx,
                    reductionOperands, reductionRecipes);
    } else if (const auto *defaultClause =
                   std::get_if<Fortran::parser::AccClause::Default>(
                       &clause.u)) {
      if ((defaultClause->v).v == llvm::acc::DefaultValue::ACC_Default_none)
        hasDefaultNone = true;
      else if ((defaultClause->v).v ==
               llvm::acc::DefaultValue::ACC_Default_present)
        hasDefaultPresent = true;
    }
  }

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value, 8> operands;
  llvm::SmallVector<int32_t, 8> operandSegments;
  addOperand(operands, operandSegments, async);
  addOperands(operands, operandSegments, waitOperands);
  if constexpr (!std::is_same_v<Op, mlir::acc::SerialOp>) {
    addOperands(operands, operandSegments, numGangs);
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

  if (hasDefaultNone)
    computeOp.setDefaultAttr(mlir::acc::ClauseDefaultValue::None);
  if (hasDefaultPresent)
    computeOp.setDefaultAttr(mlir::acc::ClauseDefaultValue::Present);

  if constexpr (!std::is_same_v<Op, mlir::acc::KernelsOp>) {
    if (!privatizations.empty())
      computeOp.setPrivatizationsAttr(
          mlir::ArrayAttr::get(builder.getContext(), privatizations));
    if (!reductionRecipes.empty())
      computeOp.setReductionRecipesAttr(
          mlir::ArrayAttr::get(builder.getContext(), reductionRecipes));
    if (!firstPrivatizations.empty())
      computeOp.setFirstprivatizationsAttr(
          mlir::ArrayAttr::get(builder.getContext(), firstPrivatizations));
  }

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
  mlir::Value ifCond, async, waitDevnum;
  llvm::SmallVector<mlir::Value> attachEntryOperands, createEntryOperands,
      copyEntryOperands, copyoutEntryOperands, dataClauseOperands, waitOperands;

  // Async and wait have an optional value but can be present with
  // no value as well. When there is no value, the op has an attribute to
  // represent the clause.
  bool addAsyncAttr = false;
  bool addWaitAttr = false;

  bool hasDefaultNone = false;
  bool hasDefaultPresent = false;

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  // Lower clauses values mapped to operands.
  // Keep track of each group of operands separately as clauses can appear
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
    } else if (const auto *asyncClause =
                   std::get_if<Fortran::parser::AccClause::Async>(&clause.u)) {
      genAsyncClause(converter, asyncClause, async, addAsyncAttr, stmtCtx);
    } else if (const auto *waitClause =
                   std::get_if<Fortran::parser::AccClause::Wait>(&clause.u)) {
      genWaitClause(converter, waitClause, waitOperands, waitDevnum,
                    addWaitAttr, stmtCtx);
    } else if(const auto *defaultClause = 
                  std::get_if<Fortran::parser::AccClause::Default>(&clause.u)) {
      if ((defaultClause->v).v == llvm::acc::DefaultValue::ACC_Default_none)
        hasDefaultNone = true;
      else if ((defaultClause->v).v == llvm::acc::DefaultValue::ACC_Default_present)
        hasDefaultPresent = true;
    }
  }

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<int32_t> operandSegments;
  addOperand(operands, operandSegments, ifCond);
  addOperand(operands, operandSegments, async);
  addOperand(operands, operandSegments, waitDevnum);
  addOperands(operands, operandSegments, waitOperands);
  addOperands(operands, operandSegments, dataClauseOperands);

  auto dataOp = createRegionOp<mlir::acc::DataOp, mlir::acc::TerminatorOp>(
      builder, currentLocation, operands, operandSegments);

  dataOp.setAsyncAttr(addAsyncAttr);
  dataOp.setWaitAttr(addWaitAttr);

  if (hasDefaultNone)
    dataOp.setDefaultAttr(mlir::acc::ClauseDefaultValue::None);
  if (hasDefaultPresent)
    dataOp.setDefaultAttr(mlir::acc::ClauseDefaultValue::Present);

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
genACCHostDataOp(Fortran::lower::AbstractConverter &converter,
                 mlir::Location currentLocation,
                 Fortran::semantics::SemanticsContext &semanticsContext,
                 Fortran::lower::StatementContext &stmtCtx,
                 const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value ifCond;
  llvm::SmallVector<mlir::Value> dataOperands;
  bool addIfPresentAttr = false;

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      genIfClause(converter, clauseLocation, ifClause, ifCond, stmtCtx);
    } else if (const auto *useDevice =
                   std::get_if<Fortran::parser::AccClause::UseDevice>(
                       &clause.u)) {
      genDataOperandOperations<mlir::acc::UseDeviceOp>(
          useDevice->v, converter, semanticsContext, stmtCtx, dataOperands,
          mlir::acc::DataClause::acc_use_device,
          /*structured=*/true);
    } else if (std::get_if<Fortran::parser::AccClause::IfPresent>(&clause.u)) {
      addIfPresentAttr = true;
    }
  }

  if (ifCond) {
    if (auto cst =
            mlir::dyn_cast<mlir::arith::ConstantOp>(ifCond.getDefiningOp()))
      if (auto boolAttr = cst.getValue().dyn_cast<mlir::BoolAttr>()) {
        if (boolAttr.getValue()) {
          // get rid of the if condition if it is always true.
          ifCond = mlir::Value();
        } else {
          // Do not generate the acc.host_data op if the if condition is always
          // false.
          return;
        }
      }
  }

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<int32_t> operandSegments;
  addOperand(operands, operandSegments, ifCond);
  addOperands(operands, operandSegments, dataOperands);

  auto hostDataOp =
      createRegionOp<mlir::acc::HostDataOp, mlir::acc::TerminatorOp>(
          builder, currentLocation, operands, operandSegments);

  if (addIfPresentAttr)
    hostDataOp.setIfPresentAttr(builder.getUnitAttr());
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
    genACCHostDataOp(converter, currentLocation, semanticsContext, stmtCtx,
                     accClauseList);
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
  // Keep track of each group of operands separately as clauses can appear
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
