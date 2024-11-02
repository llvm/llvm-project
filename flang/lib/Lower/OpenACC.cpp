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
#include "DirectivesCommon.h"
#include "flang/Common/idioms.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/IntrinsicCall.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/Frontend/OpenACC/ACC.h.inc"

// Special value for * passed in device_type or gang clauses.
static constexpr std::int64_t starCst = -1;

static unsigned routineCounter = 0;
static constexpr llvm::StringRef accRoutinePrefix = "acc_routine_";
static constexpr llvm::StringRef accPrivateInitName = "acc.private.init";
static constexpr llvm::StringRef accReductionInitName = "acc.reduction.init";
static constexpr llvm::StringRef accFirDescriptorPostfix = "_desc";

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
                            bool structured, bool implicit,
                            mlir::acc::DataClause dataClause, mlir::Type retTy,
                            mlir::Value isPresent = {}) {
  mlir::Value varPtrPtr;
  if (auto boxTy = baseAddr.getType().dyn_cast<fir::BaseBoxType>()) {
    if (isPresent) {
      baseAddr =
          builder
              .genIfOp(loc, {boxTy.getEleTy()}, isPresent,
                       /*withElseRegion=*/true)
              .genThen([&]() {
                mlir::Value boxAddr =
                    builder.create<fir::BoxAddrOp>(loc, baseAddr);
                builder.create<fir::ResultOp>(loc, mlir::ValueRange{boxAddr});
              })
              .genElse([&] {
                mlir::Value absent =
                    builder.create<fir::AbsentOp>(loc, boxTy.getEleTy());
                builder.create<fir::ResultOp>(loc, mlir::ValueRange{absent});
              })
              .getResults()[0];
    } else {
      baseAddr = builder.create<fir::BoxAddrOp>(loc, baseAddr);
    }
    retTy = baseAddr.getType();
  }

  Op op = builder.create<Op>(loc, retTy, baseAddr);
  op.setNameAttr(builder.getStringAttr(name.str()));
  op.setStructured(structured);
  op.setImplicit(implicit);
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

static void addDeclareAttr(fir::FirOpBuilder &builder, mlir::Operation *op,
                           mlir::acc::DataClause clause) {
  if (!op)
    return;
  op->setAttr(mlir::acc::getDeclareAttrName(),
              mlir::acc::DeclareAttr::get(builder.getContext(),
                                          mlir::acc::DataClauseAttr::get(
                                              builder.getContext(), clause)));
}

static mlir::func::FuncOp
createDeclareFunc(mlir::OpBuilder &modBuilder, fir::FirOpBuilder &builder,
                  mlir::Location loc, llvm::StringRef funcName,
                  llvm::SmallVector<mlir::Type> argsTy = {},
                  llvm::SmallVector<mlir::Location> locs = {}) {
  auto funcTy = mlir::FunctionType::get(modBuilder.getContext(), argsTy, {});
  auto funcOp = modBuilder.create<mlir::func::FuncOp>(loc, funcName, funcTy);
  funcOp.setVisibility(mlir::SymbolTable::Visibility::Private);
  builder.createBlock(&funcOp.getRegion(), funcOp.getRegion().end(), argsTy,
                      locs);
  builder.setInsertionPointToEnd(&funcOp.getRegion().back());
  builder.create<mlir::func::ReturnOp>(loc);
  builder.setInsertionPointToStart(&funcOp.getRegion().back());
  return funcOp;
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

template <typename EntryOp>
static void createDeclareAllocFuncWithArg(mlir::OpBuilder &modBuilder,
                                          fir::FirOpBuilder &builder,
                                          mlir::Location loc, mlir::Type descTy,
                                          llvm::StringRef funcNamePrefix,
                                          std::stringstream &asFortran,
                                          mlir::acc::DataClause clause) {
  auto crtInsPt = builder.saveInsertionPoint();
  std::stringstream registerFuncName;
  registerFuncName << funcNamePrefix.str()
                   << Fortran::lower::declarePostAllocSuffix.str();

  if (!mlir::isa<fir::ReferenceType>(descTy))
    descTy = fir::ReferenceType::get(descTy);
  auto registerFuncOp = createDeclareFunc(
      modBuilder, builder, loc, registerFuncName.str(), {descTy}, {loc});

  llvm::SmallVector<mlir::Value> bounds;
  std::stringstream asFortranDesc;
  asFortranDesc << asFortran.str() << accFirDescriptorPostfix.str();

  // Updating descriptor must occur before the mapping of the data so that
  // attached data pointer is not overwritten.
  mlir::acc::UpdateDeviceOp updateDeviceOp =
      createDataEntryOp<mlir::acc::UpdateDeviceOp>(
          builder, loc, registerFuncOp.getArgument(0), asFortranDesc, bounds,
          /*structured=*/false, /*implicit=*/true,
          mlir::acc::DataClause::acc_update_device, descTy);
  llvm::SmallVector<int32_t> operandSegments{0, 0, 0, 0, 1};
  llvm::SmallVector<mlir::Value> operands{updateDeviceOp.getResult()};
  createSimpleOp<mlir::acc::UpdateOp>(builder, loc, operands, operandSegments);

  mlir::Value desc =
      builder.create<fir::LoadOp>(loc, registerFuncOp.getArgument(0));
  fir::BoxAddrOp boxAddrOp = builder.create<fir::BoxAddrOp>(loc, desc);
  addDeclareAttr(builder, boxAddrOp.getOperation(), clause);
  EntryOp entryOp = createDataEntryOp<EntryOp>(
      builder, loc, boxAddrOp.getResult(), asFortran, bounds,
      /*structured=*/false, /*implicit=*/false, clause, boxAddrOp.getType());
  builder.create<mlir::acc::DeclareEnterOp>(
      loc, mlir::acc::DeclareTokenType::get(entryOp.getContext()),
      mlir::ValueRange(entryOp.getAccPtr()));

  modBuilder.setInsertionPointAfter(registerFuncOp);
  builder.restoreInsertionPoint(crtInsPt);
}

template <typename ExitOp>
static void createDeclareDeallocFuncWithArg(
    mlir::OpBuilder &modBuilder, fir::FirOpBuilder &builder, mlir::Location loc,
    mlir::Type descTy, llvm::StringRef funcNamePrefix,
    std::stringstream &asFortran, mlir::acc::DataClause clause) {
  auto crtInsPt = builder.saveInsertionPoint();
  // Generate the pre dealloc function.
  std::stringstream preDeallocFuncName;
  preDeallocFuncName << funcNamePrefix.str()
                     << Fortran::lower::declarePreDeallocSuffix.str();
  if (!mlir::isa<fir::ReferenceType>(descTy))
    descTy = fir::ReferenceType::get(descTy);
  auto preDeallocOp = createDeclareFunc(
      modBuilder, builder, loc, preDeallocFuncName.str(), {descTy}, {loc});
  mlir::Value loadOp =
      builder.create<fir::LoadOp>(loc, preDeallocOp.getArgument(0));
  fir::BoxAddrOp boxAddrOp = builder.create<fir::BoxAddrOp>(loc, loadOp);
  addDeclareAttr(builder, boxAddrOp.getOperation(), clause);

  llvm::SmallVector<mlir::Value> bounds;
  mlir::acc::GetDevicePtrOp entryOp =
      createDataEntryOp<mlir::acc::GetDevicePtrOp>(
          builder, loc, boxAddrOp.getResult(), asFortran, bounds,
          /*structured=*/false, /*implicit=*/false, clause,
          boxAddrOp.getType());
  builder.create<mlir::acc::DeclareExitOp>(
      loc, mlir::Value{}, mlir::ValueRange(entryOp.getAccPtr()));

  if constexpr (std::is_same_v<ExitOp, mlir::acc::CopyoutOp> ||
                std::is_same_v<ExitOp, mlir::acc::UpdateHostOp>)
    builder.create<ExitOp>(entryOp.getLoc(), entryOp.getAccPtr(),
                           entryOp.getVarPtr(), entryOp.getBounds(),
                           entryOp.getDataClause(),
                           /*structured=*/false, /*implicit=*/false,
                           builder.getStringAttr(*entryOp.getName()));
  else
    builder.create<ExitOp>(entryOp.getLoc(), entryOp.getAccPtr(),
                           entryOp.getBounds(), entryOp.getDataClause(),
                           /*structured=*/false, /*implicit=*/false,
                           builder.getStringAttr(*entryOp.getName()));

  // Generate the post dealloc function.
  modBuilder.setInsertionPointAfter(preDeallocOp);
  std::stringstream postDeallocFuncName;
  postDeallocFuncName << funcNamePrefix.str()
                      << Fortran::lower::declarePostDeallocSuffix.str();
  auto postDeallocOp = createDeclareFunc(
      modBuilder, builder, loc, postDeallocFuncName.str(), {descTy}, {loc});
  loadOp = builder.create<fir::LoadOp>(loc, postDeallocOp.getArgument(0));
  asFortran << accFirDescriptorPostfix.str();
  mlir::acc::UpdateDeviceOp updateDeviceOp =
      createDataEntryOp<mlir::acc::UpdateDeviceOp>(
          builder, loc, loadOp, asFortran, bounds,
          /*structured=*/false, /*implicit=*/true,
          mlir::acc::DataClause::acc_update_device, loadOp.getType());
  llvm::SmallVector<int32_t> operandSegments{0, 0, 0, 0, 1};
  llvm::SmallVector<mlir::Value> operands{updateDeviceOp.getResult()};
  createSimpleOp<mlir::acc::UpdateOp>(builder, loc, operands, operandSegments);
  modBuilder.setInsertionPointAfter(postDeallocOp);
  builder.restoreInsertionPoint(crtInsPt);
}

Fortran::semantics::Symbol &
getSymbolFromAccObject(const Fortran::parser::AccObject &accObject) {
  if (const auto *designator =
          std::get_if<Fortran::parser::Designator>(&accObject.u)) {
    if (const auto *name =
            Fortran::semantics::getDesignatorNameIfDataRef(*designator))
      return *name->symbol;
    if (const auto *arrayElement =
            Fortran::parser::Unwrap<Fortran::parser::ArrayElement>(
                *designator)) {
      const Fortran::parser::Name &name =
          Fortran::parser::GetLastName(arrayElement->base);
      return *name.symbol;
    }
  } else if (const auto *name =
                 std::get_if<Fortran::parser::Name>(&accObject.u)) {
    return *name->symbol;
  }
  llvm::report_fatal_error("Could not find symbol");
}

template <typename Op>
static void
genDataOperandOperations(const Fortran::parser::AccObjectList &objectList,
                         Fortran::lower::AbstractConverter &converter,
                         Fortran::semantics::SemanticsContext &semanticsContext,
                         Fortran::lower::StatementContext &stmtCtx,
                         llvm::SmallVectorImpl<mlir::Value> &dataOperands,
                         mlir::acc::DataClause dataClause, bool structured,
                         bool implicit, bool setDeclareAttr = false) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  for (const auto &accObject : objectList.v) {
    llvm::SmallVector<mlir::Value> bounds;
    std::stringstream asFortran;
    mlir::Location operandLocation = genOperandLocation(converter, accObject);
    Fortran::lower::AddrAndBoundsInfo info =
        Fortran::lower::gatherDataOperandAddrAndBounds<
            Fortran::parser::AccObject, mlir::acc::DataBoundsOp,
            mlir::acc::DataBoundsType>(converter, builder, semanticsContext,
                                       stmtCtx, accObject, operandLocation,
                                       asFortran, bounds,
                                       /*treatIndexAsSection=*/true);

    Op op = createDataEntryOp<Op>(
        builder, operandLocation, info.addr, asFortran, bounds, structured,
        implicit, dataClause, info.addr.getType(), info.isPresent);
    dataOperands.push_back(op.getAccPtr());
  }
}

template <typename EntryOp, typename ExitOp>
static void genDeclareDataOperandOperations(
    const Fortran::parser::AccObjectList &objectList,
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semanticsContext,
    Fortran::lower::StatementContext &stmtCtx,
    llvm::SmallVectorImpl<mlir::Value> &dataOperands,
    mlir::acc::DataClause dataClause, bool structured, bool implicit) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  for (const auto &accObject : objectList.v) {
    llvm::SmallVector<mlir::Value> bounds;
    std::stringstream asFortran;
    mlir::Location operandLocation = genOperandLocation(converter, accObject);
    Fortran::lower::AddrAndBoundsInfo info =
        Fortran::lower::gatherDataOperandAddrAndBounds<
            Fortran::parser::AccObject, mlir::acc::DataBoundsOp,
            mlir::acc::DataBoundsType>(converter, builder, semanticsContext,
                                       stmtCtx, accObject, operandLocation,
                                       asFortran, bounds);
    EntryOp op = createDataEntryOp<EntryOp>(
        builder, operandLocation, info.addr, asFortran, bounds, structured,
        implicit, dataClause, info.addr.getType());
    dataOperands.push_back(op.getAccPtr());
    addDeclareAttr(builder, op.getVarPtr().getDefiningOp(), dataClause);
    if (mlir::isa<fir::BaseBoxType>(fir::unwrapRefType(info.addr.getType()))) {
      mlir::OpBuilder modBuilder(builder.getModule().getBodyRegion());
      modBuilder.setInsertionPointAfter(builder.getFunction());
      std::string prefix =
          converter.mangleName(getSymbolFromAccObject(accObject));
      createDeclareAllocFuncWithArg<EntryOp>(
          modBuilder, builder, operandLocation, info.addr.getType(), prefix,
          asFortran, dataClause);
      if constexpr (!std::is_same_v<EntryOp, ExitOp>)
        createDeclareDeallocFuncWithArg<ExitOp>(
            modBuilder, builder, operandLocation, info.addr.getType(), prefix,
            asFortran, dataClause);
    }
  }
}

template <typename EntryOp, typename ExitOp, typename Clause>
static void genDeclareDataOperandOperationsWithModifier(
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
  genDeclareDataOperandOperations<EntryOp, ExitOp>(
      accObjectList, converter, semanticsContext, stmtCtx, dataClauseOperands,
      dataClause,
      /*structured=*/true, /*implicit=*/false);
}

template <typename EntryOp, typename ExitOp>
static void genDataExitOperations(fir::FirOpBuilder &builder,
                                  llvm::SmallVector<mlir::Value> operands,
                                  bool structured) {
  for (mlir::Value operand : operands) {
    auto entryOp = mlir::dyn_cast_or_null<EntryOp>(operand.getDefiningOp());
    assert(entryOp && "data entry op expected");
    if constexpr (std::is_same_v<ExitOp, mlir::acc::CopyoutOp> ||
                  std::is_same_v<ExitOp, mlir::acc::UpdateHostOp>)
      builder.create<ExitOp>(
          entryOp.getLoc(), entryOp.getAccPtr(), entryOp.getVarPtr(),
          entryOp.getBounds(), entryOp.getDataClause(), structured,
          entryOp.getImplicit(), builder.getStringAttr(*entryOp.getName()));
    else
      builder.create<ExitOp>(entryOp.getLoc(), entryOp.getAccPtr(),
                             entryOp.getBounds(), entryOp.getDataClause(),
                             structured, entryOp.getImplicit(),
                             builder.getStringAttr(*entryOp.getName()));
  }
}

fir::ShapeOp genShapeOp(mlir::OpBuilder &builder, fir::SequenceType seqTy,
                        mlir::Location loc) {
  llvm::SmallVector<mlir::Value> extents;
  mlir::Type idxTy = builder.getIndexType();
  for (auto extent : seqTy.getShape())
    extents.push_back(builder.create<mlir::arith::ConstantOp>(
        loc, idxTy, builder.getIntegerAttr(idxTy, extent)));
  return builder.create<fir::ShapeOp>(loc, extents);
}

/// Return the nested sequence type if any.
static mlir::Type extractSequenceType(mlir::Type ty) {
  if (mlir::isa<fir::SequenceType>(ty))
    return ty;
  if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(ty))
    return extractSequenceType(boxTy.getEleTy());
  if (auto heapTy = mlir::dyn_cast<fir::HeapType>(ty))
    return extractSequenceType(heapTy.getEleTy());
  if (auto ptrTy = mlir::dyn_cast<fir::PointerType>(ty))
    return extractSequenceType(ptrTy.getEleTy());
  return mlir::Type{};
}

template <typename RecipeOp>
static void genPrivateLikeInitRegion(mlir::OpBuilder &builder, RecipeOp recipe,
                                     mlir::Type ty, mlir::Location loc) {
  mlir::Value retVal = recipe.getInitRegion().front().getArgument(0);
  if (auto refTy = mlir::dyn_cast_or_null<fir::ReferenceType>(ty)) {
    if (fir::isa_trivial(refTy.getEleTy())) {
      auto alloca = builder.create<fir::AllocaOp>(loc, refTy.getEleTy());
      auto declareOp = builder.create<hlfir::DeclareOp>(
          loc, alloca, accPrivateInitName, /*shape=*/nullptr,
          llvm::ArrayRef<mlir::Value>{}, fir::FortranVariableFlagsAttr{});
      retVal = declareOp.getBase();
    } else if (auto seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(
                   refTy.getEleTy())) {
      if (fir::isa_trivial(seqTy.getEleTy())) {
        mlir::Value shape;
        llvm::SmallVector<mlir::Value> extents;
        if (seqTy.hasDynamicExtents()) {
          // Extents are passed as block arguments. First argument is the
          // original value.
          for (unsigned i = 1; i < recipe.getInitRegion().getArguments().size();
               ++i)
            extents.push_back(recipe.getInitRegion().getArgument(i));
          shape = builder.create<fir::ShapeOp>(loc, extents);
        } else {
          shape = genShapeOp(builder, seqTy, loc);
        }
        auto alloca = builder.create<fir::AllocaOp>(
            loc, seqTy, /*typeparams=*/mlir::ValueRange{}, extents);
        auto declareOp = builder.create<hlfir::DeclareOp>(
            loc, alloca, accPrivateInitName, shape,
            llvm::ArrayRef<mlir::Value>{}, fir::FortranVariableFlagsAttr{});
        retVal = declareOp.getBase();
      }
    }
  } else if (auto boxTy = mlir::dyn_cast_or_null<fir::BaseBoxType>(ty)) {
    mlir::Type innerTy = extractSequenceType(boxTy);
    if (!innerTy)
      TODO(loc, "Unsupported boxed type in OpenACC privatization");
    fir::FirOpBuilder firBuilder{builder, recipe.getOperation()};
    hlfir::Entity source = hlfir::Entity{retVal};
    auto [temp, cleanup] = hlfir::createTempFromMold(loc, firBuilder, source);
    retVal = temp;
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
  llvm::SmallVector<mlir::Type> argsTy{ty};
  llvm::SmallVector<mlir::Location> argsLoc{loc};
  if (auto refTy = mlir::dyn_cast_or_null<fir::ReferenceType>(ty)) {
    if (auto seqTy =
            mlir::dyn_cast_or_null<fir::SequenceType>(refTy.getEleTy())) {
      if (seqTy.hasDynamicExtents()) {
        mlir::Type idxTy = builder.getIndexType();
        for (unsigned i = 0; i < seqTy.getDimension(); ++i) {
          argsTy.push_back(idxTy);
          argsLoc.push_back(loc);
        }
      }
    }
  }
  builder.createBlock(&recipe.getInitRegion(), recipe.getInitRegion().end(),
                      argsTy, argsLoc);
  builder.setInsertionPointToEnd(&recipe.getInitRegion().back());
  genPrivateLikeInitRegion<mlir::acc::PrivateRecipeOp>(builder, recipe, ty,
                                                       loc);
  builder.restoreInsertionPoint(crtPos);
  return recipe;
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

/// Return true iff all the bounds are expressed with constant values.
bool areAllBoundConstant(const llvm::SmallVector<mlir::Value> &bounds) {
  for (auto bound : bounds) {
    auto dataBound =
        mlir::dyn_cast<mlir::acc::DataBoundsOp>(bound.getDefiningOp());
    assert(dataBound && "Must be DataBoundOp operation");
    if (!isConstantBound(dataBound))
      return false;
  }
  return true;
}

static llvm::SmallVector<mlir::Value>
genConstantBounds(fir::FirOpBuilder &builder, mlir::Location loc,
                  mlir::acc::DataBoundsOp &dataBound) {
  mlir::Type idxTy = builder.getIndexType();
  mlir::Value lb, ub, step;
  if (dataBound.getLowerbound() &&
      fir::getIntIfConstant(dataBound.getLowerbound()) &&
      dataBound.getUpperbound() &&
      fir::getIntIfConstant(dataBound.getUpperbound())) {
    lb = builder.createIntegerConstant(
        loc, idxTy, *fir::getIntIfConstant(dataBound.getLowerbound()));
    ub = builder.createIntegerConstant(
        loc, idxTy, *fir::getIntIfConstant(dataBound.getUpperbound()));
    step = builder.createIntegerConstant(loc, idxTy, 1);
  } else if (dataBound.getExtent()) {
    lb = builder.createIntegerConstant(loc, idxTy, 0);
    ub = builder.createIntegerConstant(
        loc, idxTy, *fir::getIntIfConstant(dataBound.getExtent()) - 1);
    step = builder.createIntegerConstant(loc, idxTy, 1);
  } else {
    llvm::report_fatal_error("Expect constant lb/ub or extent");
  }
  return {lb, ub, step};
}

static fir::ShapeOp genShapeFromBoundsOrArgs(
    mlir::Location loc, fir::FirOpBuilder &builder, fir::SequenceType seqTy,
    const llvm::SmallVector<mlir::Value> &bounds, mlir::ValueRange arguments) {
  llvm::SmallVector<mlir::Value> args;
  if (areAllBoundConstant(bounds)) {
    for (auto bound : llvm::reverse(bounds)) {
      auto dataBound =
          mlir::cast<mlir::acc::DataBoundsOp>(bound.getDefiningOp());
      args.append(genConstantBounds(builder, loc, dataBound));
    }
  } else {
    assert(((arguments.size() - 2) / 3 == seqTy.getDimension()) &&
           "Expect 3 block arguments per dimension");
    for (auto arg : arguments.drop_front(2))
      args.push_back(arg);
  }

  assert(args.size() % 3 == 0 && "Triplets must be a multiple of 3");
  llvm::SmallVector<mlir::Value> extents;
  mlir::Type idxTy = builder.getIndexType();
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  mlir::Value zero = builder.createIntegerConstant(loc, idxTy, 0);
  for (unsigned i = 0; i < args.size(); i += 3) {
    mlir::Value s1 =
        builder.create<mlir::arith::SubIOp>(loc, args[i + 1], args[0]);
    mlir::Value s2 = builder.create<mlir::arith::AddIOp>(loc, s1, one);
    mlir::Value s3 = builder.create<mlir::arith::DivSIOp>(loc, s2, args[i + 2]);
    mlir::Value cmp = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::sgt, s3, zero);
    mlir::Value ext = builder.create<mlir::arith::SelectOp>(loc, cmp, s3, zero);
    extents.push_back(ext);
  }
  return builder.create<fir::ShapeOp>(loc, extents);
}

static hlfir::DesignateOp::Subscripts
getSubscriptsFromArgs(mlir::ValueRange args) {
  hlfir::DesignateOp::Subscripts triplets;
  for (unsigned i = 2; i < args.size(); i += 3)
    triplets.emplace_back(
        hlfir::DesignateOp::Triplet{args[i], args[i + 1], args[i + 2]});
  return triplets;
}

static hlfir::Entity genDesignateWithTriplets(
    fir::FirOpBuilder &builder, mlir::Location loc, hlfir::Entity &entity,
    hlfir::DesignateOp::Subscripts &triplets, mlir::Value shape) {
  llvm::SmallVector<mlir::Value> lenParams;
  hlfir::genLengthParameters(loc, builder, entity, lenParams);
  auto designate = builder.create<hlfir::DesignateOp>(
      loc, entity.getBase().getType(), entity, /*component=*/"",
      /*componentShape=*/mlir::Value{}, triplets,
      /*substring=*/mlir::ValueRange{}, /*complexPartAttr=*/std::nullopt, shape,
      lenParams);
  return hlfir::Entity{designate.getResult()};
}

mlir::acc::FirstprivateRecipeOp Fortran::lower::createOrGetFirstprivateRecipe(
    mlir::OpBuilder &builder, llvm::StringRef recipeName, mlir::Location loc,
    mlir::Type ty, llvm::SmallVector<mlir::Value> &bounds) {
  mlir::ModuleOp mod =
      builder.getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();
  if (auto recipe =
          mod.lookupSymbol<mlir::acc::FirstprivateRecipeOp>(recipeName))
    return recipe;

  auto crtPos = builder.saveInsertionPoint();
  mlir::OpBuilder modBuilder(mod.getBodyRegion());
  auto recipe =
      modBuilder.create<mlir::acc::FirstprivateRecipeOp>(loc, recipeName, ty);
  llvm::SmallVector<mlir::Type> initArgsTy{ty};
  llvm::SmallVector<mlir::Location> initArgsLoc{loc};
  auto refTy = fir::unwrapRefType(ty);
  if (auto seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(refTy)) {
    if (seqTy.hasDynamicExtents()) {
      mlir::Type idxTy = builder.getIndexType();
      for (unsigned i = 0; i < seqTy.getDimension(); ++i) {
        initArgsTy.push_back(idxTy);
        initArgsLoc.push_back(loc);
      }
    }
  }
  builder.createBlock(&recipe.getInitRegion(), recipe.getInitRegion().end(),
                      initArgsTy, initArgsLoc);
  builder.setInsertionPointToEnd(&recipe.getInitRegion().back());
  genPrivateLikeInitRegion<mlir::acc::FirstprivateRecipeOp>(builder, recipe, ty,
                                                            loc);

  bool allConstantBound = areAllBoundConstant(bounds);
  llvm::SmallVector<mlir::Type> argsTy{ty, ty};
  llvm::SmallVector<mlir::Location> argsLoc{loc, loc};
  if (!allConstantBound) {
    for (mlir::Value bound : llvm::reverse(bounds)) {
      auto dataBound =
          mlir::dyn_cast<mlir::acc::DataBoundsOp>(bound.getDefiningOp());
      argsTy.push_back(dataBound.getLowerbound().getType());
      argsLoc.push_back(dataBound.getLowerbound().getLoc());
      argsTy.push_back(dataBound.getUpperbound().getType());
      argsLoc.push_back(dataBound.getUpperbound().getLoc());
      argsTy.push_back(dataBound.getStartIdx().getType());
      argsLoc.push_back(dataBound.getStartIdx().getLoc());
    }
  }
  builder.createBlock(&recipe.getCopyRegion(), recipe.getCopyRegion().end(),
                      argsTy, argsLoc);

  builder.setInsertionPointToEnd(&recipe.getCopyRegion().back());
  ty = fir::unwrapRefType(ty);
  if (fir::isa_trivial(ty)) {
    mlir::Value initValue = builder.create<fir::LoadOp>(
        loc, recipe.getCopyRegion().front().getArgument(0));
    builder.create<fir::StoreOp>(loc, initValue,
                                 recipe.getCopyRegion().front().getArgument(1));
  } else if (auto seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(ty)) {
    fir::FirOpBuilder firBuilder{builder, recipe.getOperation()};
    auto shape = genShapeFromBoundsOrArgs(
        loc, firBuilder, seqTy, bounds, recipe.getCopyRegion().getArguments());

    auto leftDeclOp = builder.create<hlfir::DeclareOp>(
        loc, recipe.getCopyRegion().getArgument(0), llvm::StringRef{}, shape,
        llvm::ArrayRef<mlir::Value>{}, fir::FortranVariableFlagsAttr{});
    auto rightDeclOp = builder.create<hlfir::DeclareOp>(
        loc, recipe.getCopyRegion().getArgument(1), llvm::StringRef{}, shape,
        llvm::ArrayRef<mlir::Value>{}, fir::FortranVariableFlagsAttr{});

    hlfir::DesignateOp::Subscripts triplets =
        getSubscriptsFromArgs(recipe.getCopyRegion().getArguments());
    auto leftEntity = hlfir::Entity{leftDeclOp.getBase()};
    auto left =
        genDesignateWithTriplets(firBuilder, loc, leftEntity, triplets, shape);
    auto rightEntity = hlfir::Entity{rightDeclOp.getBase()};
    auto right =
        genDesignateWithTriplets(firBuilder, loc, rightEntity, triplets, shape);

    firBuilder.create<hlfir::AssignOp>(loc, left, right);

  } else if (auto boxTy = mlir::dyn_cast_or_null<fir::BaseBoxType>(ty)) {
    fir::FirOpBuilder firBuilder{builder, recipe.getOperation()};
    llvm::SmallVector<mlir::Value> tripletArgs;
    mlir::Type innerTy = extractSequenceType(boxTy);
    fir::SequenceType seqTy =
        mlir::dyn_cast_or_null<fir::SequenceType>(innerTy);
    if (!seqTy)
      TODO(loc, "Unsupported boxed type in OpenACC firstprivate");

    auto shape = genShapeFromBoundsOrArgs(
        loc, firBuilder, seqTy, bounds, recipe.getCopyRegion().getArguments());
    hlfir::DesignateOp::Subscripts triplets =
        getSubscriptsFromArgs(recipe.getCopyRegion().getArguments());
    auto leftEntity = hlfir::Entity{recipe.getCopyRegion().getArgument(0)};
    auto left =
        genDesignateWithTriplets(firBuilder, loc, leftEntity, triplets, shape);
    auto rightEntity = hlfir::Entity{recipe.getCopyRegion().getArgument(1)};
    auto right =
        genDesignateWithTriplets(firBuilder, loc, rightEntity, triplets, shape);
    firBuilder.create<hlfir::AssignOp>(loc, left, right);
  }

  builder.create<mlir::acc::TerminatorOp>(loc);
  builder.restoreInsertionPoint(crtPos);
  return recipe;
}

/// Get a string representation of the bounds.
std::string getBoundsString(llvm::SmallVector<mlir::Value> &bounds) {
  std::stringstream boundStr;
  if (!bounds.empty())
    boundStr << "_section_";
  llvm::interleave(
      bounds,
      [&](mlir::Value bound) {
        auto boundsOp =
            mlir::cast<mlir::acc::DataBoundsOp>(bound.getDefiningOp());
        if (boundsOp.getLowerbound() &&
            fir::getIntIfConstant(boundsOp.getLowerbound()) &&
            boundsOp.getUpperbound() &&
            fir::getIntIfConstant(boundsOp.getUpperbound())) {
          boundStr << "lb" << *fir::getIntIfConstant(boundsOp.getLowerbound())
                   << ".ub" << *fir::getIntIfConstant(boundsOp.getUpperbound());
        } else if (boundsOp.getExtent() &&
                   fir::getIntIfConstant(boundsOp.getExtent())) {
          boundStr << "ext" << *fir::getIntIfConstant(boundsOp.getExtent());
        } else {
          boundStr << "?";
        }
      },
      [&] { boundStr << "x"; });
  return boundStr.str();
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
    if (mlir::isa<fir::ReferenceType, fir::PointerType>(ty))
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
    Fortran::lower::AddrAndBoundsInfo info =
        Fortran::lower::gatherDataOperandAddrAndBounds<
            Fortran::parser::AccObject, mlir::acc::DataBoundsOp,
            mlir::acc::DataBoundsType>(converter, builder, semanticsContext,
                                       stmtCtx, accObject, operandLocation,
                                       asFortran, bounds);
    RecipeOp recipe;
    mlir::Type retTy = getTypeFromBounds(bounds, info.addr.getType());
    if constexpr (std::is_same_v<RecipeOp, mlir::acc::PrivateRecipeOp>) {
      std::string recipeName =
          fir::getTypeAsString(retTy, converter.getKindMap(),
                               Fortran::lower::privatizationRecipePrefix);
      recipe = Fortran::lower::createOrGetPrivateRecipe(builder, recipeName,
                                                        operandLocation, retTy);
      auto op = createDataEntryOp<mlir::acc::PrivateOp>(
          builder, operandLocation, info.addr, asFortran, bounds, true,
          /*implicit=*/false, mlir::acc::DataClause::acc_private, retTy);
      dataOperands.push_back(op.getAccPtr());
    } else {
      std::string suffix =
          areAllBoundConstant(bounds) ? getBoundsString(bounds) : "";
      std::string recipeName = fir::getTypeAsString(
          retTy, converter.getKindMap(), "firstprivatization" + suffix);
      recipe = Fortran::lower::createOrGetFirstprivateRecipe(
          builder, recipeName, operandLocation, retTy, bounds);
      auto op = createDataEntryOp<mlir::acc::FirstprivateOp>(
          builder, operandLocation, info.addr, asFortran, bounds, true,
          /*implicit=*/false, mlir::acc::DataClause::acc_firstprivate, retTy);
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

  if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(ty))
    return getReductionInitValue(builder, loc, boxTy.getEleTy(), op);

  if (auto heapTy = mlir::dyn_cast<fir::HeapType>(ty))
    return getReductionInitValue(builder, loc, heapTy.getEleTy(), op);

  if (auto ptrTy = mlir::dyn_cast<fir::PointerType>(ty))
    return getReductionInitValue(builder, loc, ptrTy.getEleTy(), op);

  llvm::report_fatal_error("Unsupported OpenACC reduction type");
}

static mlir::Value genReductionInitRegion(fir::FirOpBuilder &builder,
                                          mlir::Location loc, mlir::Type ty,
                                          mlir::acc::ReductionOperator op) {
  ty = fir::unwrapRefType(ty);
  mlir::Value initValue = getReductionInitValue(builder, loc, ty, op);
  if (fir::isa_trivial(ty)) {
    mlir::Value alloca = builder.create<fir::AllocaOp>(loc, ty);
    auto declareOp = builder.create<hlfir::DeclareOp>(
        loc, alloca, accReductionInitName, /*shape=*/nullptr,
        llvm::ArrayRef<mlir::Value>{}, fir::FortranVariableFlagsAttr{});
    builder.create<fir::StoreOp>(loc, builder.createConvert(loc, ty, initValue),
                                 declareOp.getBase());
    return declareOp.getBase();
  } else if (auto seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(ty)) {
    if (fir::isa_trivial(seqTy.getEleTy())) {
      mlir::Value shape;
      auto extents = builder.getBlock()->getArguments().drop_front(1);
      if (seqTy.hasDynamicExtents())
        shape = builder.create<fir::ShapeOp>(loc, extents);
      else
        shape = genShapeOp(builder, seqTy, loc);
      mlir::Value alloca = builder.create<fir::AllocaOp>(
          loc, seqTy, /*typeparams=*/mlir::ValueRange{}, extents);
      auto declareOp = builder.create<hlfir::DeclareOp>(
          loc, alloca, accReductionInitName, shape,
          llvm::ArrayRef<mlir::Value>{}, fir::FortranVariableFlagsAttr{});
      mlir::Type idxTy = builder.getIndexType();
      mlir::Type refTy = fir::ReferenceType::get(seqTy.getEleTy());
      llvm::SmallVector<fir::DoLoopOp> loops;
      llvm::SmallVector<mlir::Value> ivs;

      if (seqTy.hasDynamicExtents()) {
        builder.create<hlfir::AssignOp>(loc, initValue, declareOp.getBase());
        return declareOp.getBase();
      }
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
      auto coord = builder.create<fir::CoordinateOp>(loc, refTy,
                                                     declareOp.getBase(), ivs);
      builder.create<fir::StoreOp>(loc, initValue, coord);
      builder.setInsertionPointAfter(loops[0]);
      return declareOp.getBase();
    }
  } else if (auto boxTy = mlir::dyn_cast_or_null<fir::BaseBoxType>(ty)) {
    mlir::Type innerTy = extractSequenceType(boxTy);
    if (!mlir::isa<fir::SequenceType>(innerTy))
      TODO(loc, "Unsupported boxed type for reduction");
    // Create the private copy from the initial fir.box.
    hlfir::Entity source = hlfir::Entity{builder.getBlock()->getArgument(0)};
    auto [temp, cleanup] = hlfir::createTempFromMold(loc, builder, source);
    builder.create<hlfir::AssignOp>(loc, initValue, temp);
    return temp;
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

static hlfir::DesignateOp::Subscripts
getTripletsFromArgs(mlir::acc::ReductionRecipeOp recipe) {
  hlfir::DesignateOp::Subscripts triplets;
  for (unsigned i = 2; i < recipe.getCombinerRegion().getArguments().size();
       i += 3)
    triplets.emplace_back(hlfir::DesignateOp::Triplet{
        recipe.getCombinerRegion().getArgument(i),
        recipe.getCombinerRegion().getArgument(i + 1),
        recipe.getCombinerRegion().getArgument(i + 2)});
  return triplets;
}

static void genCombiner(fir::FirOpBuilder &builder, mlir::Location loc,
                        mlir::acc::ReductionOperator op, mlir::Type ty,
                        mlir::Value value1, mlir::Value value2,
                        mlir::acc::ReductionRecipeOp &recipe,
                        llvm::SmallVector<mlir::Value> &bounds,
                        bool allConstantBound) {
  ty = fir::unwrapRefType(ty);

  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(ty)) {
    mlir::Type refTy = fir::ReferenceType::get(seqTy.getEleTy());
    llvm::SmallVector<fir::DoLoopOp> loops;
    llvm::SmallVector<mlir::Value> ivs;
    if (seqTy.hasDynamicExtents()) {
      auto shape =
          genShapeFromBoundsOrArgs(loc, builder, seqTy, bounds,
                                   recipe.getCombinerRegion().getArguments());
      auto v1DeclareOp = builder.create<hlfir::DeclareOp>(
          loc, value1, llvm::StringRef{}, shape, llvm::ArrayRef<mlir::Value>{},
          fir::FortranVariableFlagsAttr{});
      auto v2DeclareOp = builder.create<hlfir::DeclareOp>(
          loc, value2, llvm::StringRef{}, shape, llvm::ArrayRef<mlir::Value>{},
          fir::FortranVariableFlagsAttr{});
      hlfir::DesignateOp::Subscripts triplets = getTripletsFromArgs(recipe);

      llvm::SmallVector<mlir::Value> lenParamsLeft;
      auto leftEntity = hlfir::Entity{v1DeclareOp.getBase()};
      hlfir::genLengthParameters(loc, builder, leftEntity, lenParamsLeft);
      auto leftDesignate = builder.create<hlfir::DesignateOp>(
          loc, v1DeclareOp.getBase().getType(), v1DeclareOp.getBase(),
          /*component=*/"",
          /*componentShape=*/mlir::Value{}, triplets,
          /*substring=*/mlir::ValueRange{}, /*complexPartAttr=*/std::nullopt,
          shape, lenParamsLeft);
      auto left = hlfir::Entity{leftDesignate.getResult()};

      llvm::SmallVector<mlir::Value> lenParamsRight;
      auto rightEntity = hlfir::Entity{v2DeclareOp.getBase()};
      hlfir::genLengthParameters(loc, builder, rightEntity, lenParamsLeft);
      auto rightDesignate = builder.create<hlfir::DesignateOp>(
          loc, v2DeclareOp.getBase().getType(), v2DeclareOp.getBase(),
          /*component=*/"",
          /*componentShape=*/mlir::Value{}, triplets,
          /*substring=*/mlir::ValueRange{}, /*complexPartAttr=*/std::nullopt,
          shape, lenParamsRight);
      auto right = hlfir::Entity{rightDesignate.getResult()};

      llvm::SmallVector<mlir::Value, 1> typeParams;
      auto genKernel = [&builder, &loc, op, seqTy, &left, &right](
                           mlir::Location l, fir::FirOpBuilder &b,
                           mlir::ValueRange oneBasedIndices) -> hlfir::Entity {
        auto leftElement = hlfir::getElementAt(l, b, left, oneBasedIndices);
        auto rightElement = hlfir::getElementAt(l, b, right, oneBasedIndices);
        auto leftVal = hlfir::loadTrivialScalar(l, b, leftElement);
        auto rightVal = hlfir::loadTrivialScalar(l, b, rightElement);
        return hlfir::Entity{genScalarCombiner(
            builder, loc, op, seqTy.getEleTy(), leftVal, rightVal)};
      };
      mlir::Value elemental = hlfir::genElementalOp(
          loc, builder, seqTy.getEleTy(), shape, typeParams, genKernel,
          /*isUnordered=*/true);
      builder.create<hlfir::AssignOp>(loc, elemental, v1DeclareOp.getBase());
      return;
    }
    if (allConstantBound) {
      // Use the constant bound directly in the combiner region so they do not
      // need to be passed as block argument.
      for (auto bound : llvm::reverse(bounds)) {
        auto dataBound =
            mlir::dyn_cast<mlir::acc::DataBoundsOp>(bound.getDefiningOp());
        llvm::SmallVector<mlir::Value> values =
            genConstantBounds(builder, loc, dataBound);
        auto loop =
            builder.create<fir::DoLoopOp>(loc, values[0], values[1], values[2],
                                          /*unordered=*/false);
        builder.setInsertionPointToStart(loop.getBody());
        loops.push_back(loop);
        ivs.push_back(loop.getInductionVar());
      }
    } else {
      // Lowerbound, upperbound and step are passed as block arguments.
      [[maybe_unused]] unsigned nbRangeArgs =
          recipe.getCombinerRegion().getArguments().size() - 2;
      assert((nbRangeArgs / 3 == seqTy.getDimension()) &&
             "Expect 3 block arguments per dimension");
      for (unsigned i = 2; i < recipe.getCombinerRegion().getArguments().size();
           i += 3) {
        mlir::Value lb = recipe.getCombinerRegion().getArgument(i);
        mlir::Value ub = recipe.getCombinerRegion().getArgument(i + 1);
        mlir::Value step = recipe.getCombinerRegion().getArgument(i + 2);
        auto loop = builder.create<fir::DoLoopOp>(loc, lb, ub, step,
                                                  /*unordered=*/false);
        builder.setInsertionPointToStart(loop.getBody());
        loops.push_back(loop);
        ivs.push_back(loop.getInductionVar());
      }
    }
    auto addr1 = builder.create<fir::CoordinateOp>(loc, refTy, value1, ivs);
    auto addr2 = builder.create<fir::CoordinateOp>(loc, refTy, value2, ivs);
    auto load1 = builder.create<fir::LoadOp>(loc, addr1);
    auto load2 = builder.create<fir::LoadOp>(loc, addr2);
    mlir::Value res =
        genScalarCombiner(builder, loc, op, seqTy.getEleTy(), load1, load2);
    builder.create<fir::StoreOp>(loc, res, addr1);
    builder.setInsertionPointAfter(loops[0]);
  } else if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(ty)) {
    mlir::Type innerTy = extractSequenceType(boxTy);
    fir::SequenceType seqTy =
        mlir::dyn_cast_or_null<fir::SequenceType>(innerTy);
    if (!seqTy)
      TODO(loc, "Unsupported boxed type in OpenACC reduction");

    auto shape = genShapeFromBoundsOrArgs(
        loc, builder, seqTy, bounds, recipe.getCombinerRegion().getArguments());
    hlfir::DesignateOp::Subscripts triplets =
        getSubscriptsFromArgs(recipe.getCombinerRegion().getArguments());
    auto leftEntity = hlfir::Entity{value1};
    auto left =
        genDesignateWithTriplets(builder, loc, leftEntity, triplets, shape);
    auto rightEntity = hlfir::Entity{value2};
    auto right =
        genDesignateWithTriplets(builder, loc, rightEntity, triplets, shape);

    llvm::SmallVector<mlir::Value, 1> typeParams;
    auto genKernel = [&builder, &loc, op, seqTy, &left, &right](
                         mlir::Location l, fir::FirOpBuilder &b,
                         mlir::ValueRange oneBasedIndices) -> hlfir::Entity {
      auto leftElement = hlfir::getElementAt(l, b, left, oneBasedIndices);
      auto rightElement = hlfir::getElementAt(l, b, right, oneBasedIndices);
      auto leftVal = hlfir::loadTrivialScalar(l, b, leftElement);
      auto rightVal = hlfir::loadTrivialScalar(l, b, rightElement);
      return hlfir::Entity{genScalarCombiner(builder, loc, op, seqTy.getEleTy(),
                                             leftVal, rightVal)};
    };
    mlir::Value elemental = hlfir::genElementalOp(
        loc, builder, seqTy.getEleTy(), shape, typeParams, genKernel,
        /*isUnordered=*/true);
    builder.create<hlfir::AssignOp>(loc, elemental, value1);
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
  llvm::SmallVector<mlir::Type> initArgsTy{ty};
  llvm::SmallVector<mlir::Location> initArgsLoc{loc};
  mlir::Type refTy = fir::unwrapRefType(ty);
  if (auto seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(refTy)) {
    if (seqTy.hasDynamicExtents()) {
      mlir::Type idxTy = builder.getIndexType();
      for (unsigned i = 0; i < seqTy.getDimension(); ++i) {
        initArgsTy.push_back(idxTy);
        initArgsLoc.push_back(loc);
      }
    }
  }
  builder.createBlock(&recipe.getInitRegion(), recipe.getInitRegion().end(),
                      initArgsTy, initArgsLoc);
  builder.setInsertionPointToEnd(&recipe.getInitRegion().back());
  mlir::Value initValue = genReductionInitRegion(builder, loc, ty, op);
  builder.create<mlir::acc::YieldOp>(loc, initValue);

  // The two first block arguments are the two values to be combined.
  // The next arguments are the iteration ranges (lb, ub, step) to be used
  // for the combiner if needed.
  llvm::SmallVector<mlir::Type> argsTy{ty, ty};
  llvm::SmallVector<mlir::Location> argsLoc{loc, loc};
  bool allConstantBound = areAllBoundConstant(bounds);
  if (!allConstantBound) {
    for (mlir::Value bound : llvm::reverse(bounds)) {
      auto dataBound =
          mlir::dyn_cast<mlir::acc::DataBoundsOp>(bound.getDefiningOp());
      argsTy.push_back(dataBound.getLowerbound().getType());
      argsLoc.push_back(dataBound.getLowerbound().getLoc());
      argsTy.push_back(dataBound.getUpperbound().getType());
      argsLoc.push_back(dataBound.getUpperbound().getLoc());
      argsTy.push_back(dataBound.getStartIdx().getType());
      argsLoc.push_back(dataBound.getStartIdx().getLoc());
    }
  }
  builder.createBlock(&recipe.getCombinerRegion(),
                      recipe.getCombinerRegion().end(), argsTy, argsLoc);
  builder.setInsertionPointToEnd(&recipe.getCombinerRegion().back());
  mlir::Value v1 = recipe.getCombinerRegion().front().getArgument(0);
  mlir::Value v2 = recipe.getCombinerRegion().front().getArgument(1);
  genCombiner(builder, loc, op, ty, v1, v2, recipe, bounds, allConstantBound);
  builder.create<mlir::acc::YieldOp>(loc, v1);
  builder.restoreInsertionPoint(crtPos);
  return recipe;
}

static bool isSupportedReductionType(mlir::Type ty) {
  ty = fir::unwrapRefType(ty);
  if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(ty))
    return isSupportedReductionType(boxTy.getEleTy());
  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(ty))
    return isSupportedReductionType(seqTy.getEleTy());
  if (auto heapTy = mlir::dyn_cast<fir::HeapType>(ty))
    return isSupportedReductionType(heapTy.getEleTy());
  if (auto ptrTy = mlir::dyn_cast<fir::PointerType>(ty))
    return isSupportedReductionType(ptrTy.getEleTy());
  return fir::isa_trivial(ty);
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
    Fortran::lower::AddrAndBoundsInfo info =
        Fortran::lower::gatherDataOperandAddrAndBounds<
            Fortran::parser::AccObject, mlir::acc::DataBoundsOp,
            mlir::acc::DataBoundsType>(converter, builder, semanticsContext,
                                       stmtCtx, accObject, operandLocation,
                                       asFortran, bounds);

    mlir::Type reductionTy = fir::unwrapRefType(info.addr.getType());
    if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(reductionTy))
      reductionTy = seqTy.getEleTy();

    if (!isSupportedReductionType(reductionTy))
      TODO(operandLocation, "reduction with unsupported type");

    auto op = createDataEntryOp<mlir::acc::ReductionOp>(
        builder, operandLocation, info.addr, asFortran, bounds,
        /*structured=*/true, /*implicit=*/false,
        mlir::acc::DataClause::acc_reduction, info.addr.getType());
    mlir::Type ty = op.getAccPtr().getType();
    if (!areAllBoundConstant(bounds) ||
        fir::isAssumedShape(info.addr.getType()) ||
        fir::isAllocatableOrPointerArray(info.addr.getType()))
      ty = info.addr.getType();
    std::string suffix =
        areAllBoundConstant(bounds) ? getBoundsString(bounds) : "";
    std::string recipeName = fir::getTypeAsString(
        ty, converter.getKindMap(),
        ("reduction_" + stringifyReductionOperator(mlirOp)).str() + suffix);

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
               mlir::Location returnLoc, Fortran::lower::pft::Evaluation &eval,
               const llvm::SmallVectorImpl<mlir::Value> &operands,
               const llvm::SmallVectorImpl<int32_t> &operandSegments,
               bool outerCombined = false,
               llvm::SmallVector<mlir::Type> retTy = {},
               mlir::Value yieldValue = {}, mlir::TypeRange argsTy = {},
               llvm::SmallVector<mlir::Location> locs = {}) {
  Op op = builder.create<Op>(loc, retTy, operands);
  builder.createBlock(&op.getRegion(), op.getRegion().end(), argsTy, locs);
  mlir::Block &block = op.getRegion().back();
  builder.setInsertionPointToStart(&block);

  op->setAttr(Op::getOperandSegmentSizeAttr(),
              builder.getDenseI32ArrayAttr(operandSegments));

  // Place the insertion point to the start of the first block.
  builder.setInsertionPointToStart(&block);

  // If it is an unstructured region and is not the outer region of a combined
  // construct, create empty blocks for all evaluations.
  if (eval.lowerAsUnstructured() && !outerCombined)
    Fortran::lower::createEmptyRegionBlocks<mlir::acc::TerminatorOp,
                                            mlir::acc::YieldOp>(
        builder, eval.getNestedEvaluations());

  if (yieldValue) {
    if constexpr (std::is_same_v<Terminator, mlir::acc::YieldOp>) {
      Terminator yieldOp = builder.create<Terminator>(returnLoc, yieldValue);
      yieldValue.getDefiningOp()->moveBefore(yieldOp);
    } else {
      builder.create<Terminator>(returnLoc);
    }
  } else {
    builder.create<Terminator>(returnLoc);
  }
  builder.setInsertionPointToStart(&block);
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

static void
genAsyncClause(Fortran::lower::AbstractConverter &converter,
               const Fortran::parser::AccClause::Async *asyncClause,
               llvm::SmallVector<mlir::Value> &async,
               llvm::SmallVector<mlir::Attribute> &asyncDeviceTypes,
               llvm::SmallVector<mlir::Attribute> &asyncOnlyDeviceTypes,
               llvm::SmallVector<mlir::Attribute> &deviceTypeAttrs,
               Fortran::lower::StatementContext &stmtCtx) {
  const auto &asyncClauseValue = asyncClause->v;
  if (asyncClauseValue) { // async has a value.
    mlir::Value asyncValue = fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(*asyncClauseValue), stmtCtx));
    for (auto deviceTypeAttr : deviceTypeAttrs) {
      async.push_back(asyncValue);
      asyncDeviceTypes.push_back(deviceTypeAttr);
    }
  } else {
    for (auto deviceTypeAttr : deviceTypeAttrs)
      asyncOnlyDeviceTypes.push_back(deviceTypeAttr);
  }
}

static mlir::acc::DeviceType
getDeviceType(Fortran::common::OpenACCDeviceType device) {
  switch (device) {
  case Fortran::common::OpenACCDeviceType::Star:
    return mlir::acc::DeviceType::Star;
  case Fortran::common::OpenACCDeviceType::Default:
    return mlir::acc::DeviceType::Default;
  case Fortran::common::OpenACCDeviceType::Nvidia:
    return mlir::acc::DeviceType::Nvidia;
  case Fortran::common::OpenACCDeviceType::Radeon:
    return mlir::acc::DeviceType::Radeon;
  case Fortran::common::OpenACCDeviceType::Host:
    return mlir::acc::DeviceType::Host;
  case Fortran::common::OpenACCDeviceType::Multicore:
    return mlir::acc::DeviceType::Multicore;
  case Fortran::common::OpenACCDeviceType::None:
    return mlir::acc::DeviceType::None;
  }
  return mlir::acc::DeviceType::None;
}

static void gatherDeviceTypeAttrs(
    fir::FirOpBuilder &builder,
    const Fortran::parser::AccClause::DeviceType *deviceTypeClause,
    llvm::SmallVector<mlir::Attribute> &deviceTypes) {
  const Fortran::parser::AccDeviceTypeExprList &deviceTypeExprList =
      deviceTypeClause->v;
  for (const auto &deviceTypeExpr : deviceTypeExprList.v)
    deviceTypes.push_back(mlir::acc::DeviceTypeAttr::get(
        builder.getContext(), getDeviceType(deviceTypeExpr.v)));
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

static void
genWaitClause(Fortran::lower::AbstractConverter &converter,
              const Fortran::parser::AccClause::Wait *waitClause,
              llvm::SmallVector<mlir::Value> &waitOperands,
              llvm::SmallVector<mlir::Attribute> &waitOperandsDeviceTypes,
              llvm::SmallVector<mlir::Attribute> &waitOnlyDeviceTypes,
              llvm::SmallVector<int32_t> &waitOperandsSegments,
              mlir::Value &waitDevnum,
              llvm::SmallVector<mlir::Attribute> deviceTypeAttrs,
              Fortran::lower::StatementContext &stmtCtx) {
  const auto &waitClauseValue = waitClause->v;
  if (waitClauseValue) { // wait has a value.
    const Fortran::parser::AccWaitArgument &waitArg = *waitClauseValue;
    const auto &waitList =
        std::get<std::list<Fortran::parser::ScalarIntExpr>>(waitArg.t);
    llvm::SmallVector<mlir::Value> waitValues;
    for (const Fortran::parser::ScalarIntExpr &value : waitList) {
      waitValues.push_back(fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(value), stmtCtx)));
    }
    for (auto deviceTypeAttr : deviceTypeAttrs) {
      for (auto value : waitValues)
        waitOperands.push_back(value);
      waitOperandsDeviceTypes.push_back(deviceTypeAttr);
      waitOperandsSegments.push_back(waitValues.size());
    }

    // TODO: move to device_type model.
    const auto &waitDevnumValue =
        std::get<std::optional<Fortran::parser::ScalarIntExpr>>(waitArg.t);
    if (waitDevnumValue)
      waitDevnum = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(*waitDevnumValue), stmtCtx));
  } else {
    for (auto deviceTypeAttr : deviceTypeAttrs)
      waitOnlyDeviceTypes.push_back(deviceTypeAttr);
  }
}

mlir::Type getTypeFromIvTypeSize(fir::FirOpBuilder &builder,
                                 const Fortran::semantics::Symbol &ivSym) {
  std::size_t ivTypeSize = ivSym.size();
  if (ivTypeSize == 0)
    llvm::report_fatal_error("unexpected induction variable size");
  // ivTypeSize is in bytes and IntegerType needs to be in bits.
  return builder.getIntegerType(ivTypeSize * 8);
}

static mlir::acc::LoopOp
createLoopOp(Fortran::lower::AbstractConverter &converter,
             mlir::Location currentLocation,
             Fortran::semantics::SemanticsContext &semanticsContext,
             Fortran::lower::StatementContext &stmtCtx,
             const Fortran::parser::DoConstruct &outerDoConstruct,
             Fortran::lower::pft::Evaluation &eval,
             const Fortran::parser::AccClauseList &accClauseList,
             bool needEarlyReturnHandling = false) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  llvm::SmallVector<mlir::Value> tileOperands, privateOperands, ivPrivate,
      reductionOperands, cacheOperands, vectorOperands, workerNumOperands,
      gangOperands, lowerbounds, upperbounds, steps;
  llvm::SmallVector<mlir::Attribute> privatizations, reductionRecipes;
  llvm::SmallVector<int32_t> tileOperandsSegments, gangOperandsSegments;
  llvm::SmallVector<int64_t> collapseValues;

  llvm::SmallVector<mlir::Attribute> gangArgTypes;
  llvm::SmallVector<mlir::Attribute> seqDeviceTypes, independentDeviceTypes,
      autoDeviceTypes, vectorOperandsDeviceTypes, workerNumOperandsDeviceTypes,
      vectorDeviceTypes, workerNumDeviceTypes, tileOperandsDeviceTypes,
      collapseDeviceTypes, gangDeviceTypes, gangOperandsDeviceTypes;

  // device_type attribute is set to `none` until a device_type clause is
  // encountered.
  llvm::SmallVector<mlir::Attribute> crtDeviceTypes;
  crtDeviceTypes.push_back(mlir::acc::DeviceTypeAttr::get(
      builder.getContext(), mlir::acc::DeviceType::None));

  llvm::SmallVector<mlir::Type> ivTypes;
  llvm::SmallVector<mlir::Location> ivLocs;
  llvm::SmallVector<bool> inclusiveBounds;

  if (outerDoConstruct.IsDoConcurrent())
    TODO(currentLocation, "OpenACC loop with DO CONCURRENT");

  llvm::SmallVector<mlir::Location> locs;
  locs.push_back(currentLocation); // Location of the directive

  int64_t collapseValue = Fortran::lower::getCollapseValue(accClauseList);
  Fortran::lower::pft::Evaluation *crtEval = &eval.getFirstNestedEvaluation();
  for (unsigned i = 0; i < collapseValue; ++i) {
    const Fortran::parser::LoopControl *loopControl;
    if (i == 0) {
      loopControl = &*outerDoConstruct.GetLoopControl();
      locs.push_back(converter.genLocation(
          Fortran::parser::FindSourceLocation(outerDoConstruct)));
    } else {
      auto *doCons = crtEval->getIf<Fortran::parser::DoConstruct>();
      assert(doCons && "expect do construct");
      loopControl = &*doCons->GetLoopControl();
      locs.push_back(
          converter.genLocation(Fortran::parser::FindSourceLocation(*doCons)));
    }

    const Fortran::parser::LoopControl::Bounds *bounds =
        std::get_if<Fortran::parser::LoopControl::Bounds>(&loopControl->u);
    assert(bounds && "Expected bounds on the loop construct");
    lowerbounds.push_back(fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(bounds->lower), stmtCtx)));
    upperbounds.push_back(fir::getBase(converter.genExprValue(
        *Fortran::semantics::GetExpr(bounds->upper), stmtCtx)));
    if (bounds->step)
      steps.push_back(fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(bounds->step), stmtCtx)));
    else // If `step` is not present, assume it as `1`.
      steps.push_back(builder.createIntegerConstant(
          currentLocation, upperbounds[upperbounds.size() - 1].getType(), 1));

    Fortran::semantics::Symbol &ivSym =
        bounds->name.thing.symbol->GetUltimate();

    mlir::Type ivTy = getTypeFromIvTypeSize(builder, ivSym);
    mlir::Value ivValue = converter.getSymbolAddress(ivSym);
    ivTypes.push_back(ivTy);
    ivLocs.push_back(currentLocation);
    std::string recipeName =
        fir::getTypeAsString(ivValue.getType(), converter.getKindMap(),
                             Fortran::lower::privatizationRecipePrefix);
    auto recipe = Fortran::lower::createOrGetPrivateRecipe(
        builder, recipeName, currentLocation, ivValue.getType());
    std::stringstream asFortran;
    auto op = createDataEntryOp<mlir::acc::PrivateOp>(
        builder, currentLocation, ivValue, asFortran, {}, true,
        /*implicit=*/true, mlir::acc::DataClause::acc_private,
        ivValue.getType());

    privateOperands.push_back(op.getAccPtr());
    ivPrivate.push_back(op.getAccPtr());
    privatizations.push_back(mlir::SymbolRefAttr::get(
        builder.getContext(), recipe.getSymName().str()));
    inclusiveBounds.push_back(true);
    converter.bindSymbol(ivSym, op.getAccPtr());
    if (i < collapseValue - 1)
      crtEval = &*std::next(crtEval->getNestedEvaluations().begin());
  }

  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *gangClause =
            std::get_if<Fortran::parser::AccClause::Gang>(&clause.u)) {
      if (gangClause->v) {
        const Fortran::parser::AccGangArgList &x = *gangClause->v;
        mlir::SmallVector<mlir::Value> gangValues;
        mlir::SmallVector<mlir::Attribute> gangArgs;
        for (const Fortran::parser::AccGangArg &gangArg : x.v) {
          if (const auto *num =
                  std::get_if<Fortran::parser::AccGangArg::Num>(&gangArg.u)) {
            gangValues.push_back(fir::getBase(converter.genExprValue(
                *Fortran::semantics::GetExpr(num->v), stmtCtx)));
            gangArgs.push_back(mlir::acc::GangArgTypeAttr::get(
                builder.getContext(), mlir::acc::GangArgType::Num));
          } else if (const auto *staticArg =
                         std::get_if<Fortran::parser::AccGangArg::Static>(
                             &gangArg.u)) {
            const Fortran::parser::AccSizeExpr &sizeExpr = staticArg->v;
            if (sizeExpr.v) {
              gangValues.push_back(fir::getBase(converter.genExprValue(
                  *Fortran::semantics::GetExpr(*sizeExpr.v), stmtCtx)));
            } else {
              // * was passed as value and will be represented as a special
              // constant.
              gangValues.push_back(builder.createIntegerConstant(
                  clauseLocation, builder.getIndexType(), starCst));
            }
            gangArgs.push_back(mlir::acc::GangArgTypeAttr::get(
                builder.getContext(), mlir::acc::GangArgType::Static));
          } else if (const auto *dim =
                         std::get_if<Fortran::parser::AccGangArg::Dim>(
                             &gangArg.u)) {
            gangValues.push_back(fir::getBase(converter.genExprValue(
                *Fortran::semantics::GetExpr(dim->v), stmtCtx)));
            gangArgs.push_back(mlir::acc::GangArgTypeAttr::get(
                builder.getContext(), mlir::acc::GangArgType::Dim));
          }
        }
        for (auto crtDeviceTypeAttr : crtDeviceTypes) {
          for (const auto &pair : llvm::zip(gangValues, gangArgs)) {
            gangOperands.push_back(std::get<0>(pair));
            gangArgTypes.push_back(std::get<1>(pair));
          }
          gangOperandsSegments.push_back(gangValues.size());
          gangOperandsDeviceTypes.push_back(crtDeviceTypeAttr);
        }
      } else {
        for (auto crtDeviceTypeAttr : crtDeviceTypes)
          gangDeviceTypes.push_back(crtDeviceTypeAttr);
      }
    } else if (const auto *workerClause =
                   std::get_if<Fortran::parser::AccClause::Worker>(&clause.u)) {
      if (workerClause->v) {
        mlir::Value workerNumValue = fir::getBase(converter.genExprValue(
            *Fortran::semantics::GetExpr(*workerClause->v), stmtCtx));
        for (auto crtDeviceTypeAttr : crtDeviceTypes) {
          workerNumOperands.push_back(workerNumValue);
          workerNumOperandsDeviceTypes.push_back(crtDeviceTypeAttr);
        }
      } else {
        for (auto crtDeviceTypeAttr : crtDeviceTypes)
          workerNumDeviceTypes.push_back(crtDeviceTypeAttr);
      }
    } else if (const auto *vectorClause =
                   std::get_if<Fortran::parser::AccClause::Vector>(&clause.u)) {
      if (vectorClause->v) {
        mlir::Value vectorValue = fir::getBase(converter.genExprValue(
            *Fortran::semantics::GetExpr(*vectorClause->v), stmtCtx));
        for (auto crtDeviceTypeAttr : crtDeviceTypes) {
          vectorOperands.push_back(vectorValue);
          vectorOperandsDeviceTypes.push_back(crtDeviceTypeAttr);
        }
      } else {
        for (auto crtDeviceTypeAttr : crtDeviceTypes)
          vectorDeviceTypes.push_back(crtDeviceTypeAttr);
      }
    } else if (const auto *tileClause =
                   std::get_if<Fortran::parser::AccClause::Tile>(&clause.u)) {
      const Fortran::parser::AccTileExprList &accTileExprList = tileClause->v;
      llvm::SmallVector<mlir::Value> tileValues;
      for (const auto &accTileExpr : accTileExprList.v) {
        const auto &expr =
            std::get<std::optional<Fortran::parser::ScalarIntConstantExpr>>(
                accTileExpr.t);
        if (expr) {
          tileValues.push_back(fir::getBase(converter.genExprValue(
              *Fortran::semantics::GetExpr(*expr), stmtCtx)));
        } else {
          // * was passed as value and will be represented as a special
          // constant.
          mlir::Value tileStar = builder.createIntegerConstant(
              clauseLocation, builder.getIntegerType(32), starCst);
          tileValues.push_back(tileStar);
        }
      }
      for (auto crtDeviceTypeAttr : crtDeviceTypes) {
        for (auto value : tileValues)
          tileOperands.push_back(value);
        tileOperandsDeviceTypes.push_back(crtDeviceTypeAttr);
        tileOperandsSegments.push_back(tileValues.size());
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
    } else if (std::get_if<Fortran::parser::AccClause::Seq>(&clause.u)) {
      for (auto crtDeviceTypeAttr : crtDeviceTypes)
        seqDeviceTypes.push_back(crtDeviceTypeAttr);
    } else if (std::get_if<Fortran::parser::AccClause::Independent>(
                   &clause.u)) {
      for (auto crtDeviceTypeAttr : crtDeviceTypes)
        independentDeviceTypes.push_back(crtDeviceTypeAttr);
    } else if (std::get_if<Fortran::parser::AccClause::Auto>(&clause.u)) {
      for (auto crtDeviceTypeAttr : crtDeviceTypes)
        autoDeviceTypes.push_back(crtDeviceTypeAttr);
    } else if (const auto *deviceTypeClause =
                   std::get_if<Fortran::parser::AccClause::DeviceType>(
                       &clause.u)) {
      crtDeviceTypes.clear();
      gatherDeviceTypeAttrs(builder, deviceTypeClause, crtDeviceTypes);
    } else if (const auto *collapseClause =
                   std::get_if<Fortran::parser::AccClause::Collapse>(
                       &clause.u)) {
      const Fortran::parser::AccCollapseArg &arg = collapseClause->v;
      const auto &force = std::get<bool>(arg.t);
      if (force)
        TODO(clauseLocation, "OpenACC collapse force modifier");

      const auto &intExpr =
          std::get<Fortran::parser::ScalarIntConstantExpr>(arg.t);
      const auto *expr = Fortran::semantics::GetExpr(intExpr);
      const std::optional<int64_t> collapseValue =
          Fortran::evaluate::ToInt64(*expr);
      assert(collapseValue && "expect integer value for the collapse clause");

      for (auto crtDeviceTypeAttr : crtDeviceTypes) {
        collapseValues.push_back(*collapseValue);
        collapseDeviceTypes.push_back(crtDeviceTypeAttr);
      }
    }
  }

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<int32_t> operandSegments;
  addOperands(operands, operandSegments, lowerbounds);
  addOperands(operands, operandSegments, upperbounds);
  addOperands(operands, operandSegments, steps);
  addOperands(operands, operandSegments, gangOperands);
  addOperands(operands, operandSegments, workerNumOperands);
  addOperands(operands, operandSegments, vectorOperands);
  addOperands(operands, operandSegments, tileOperands);
  addOperands(operands, operandSegments, cacheOperands);
  addOperands(operands, operandSegments, privateOperands);
  addOperands(operands, operandSegments, reductionOperands);

  llvm::SmallVector<mlir::Type> retTy;
  mlir::Value yieldValue;
  if (needEarlyReturnHandling) {
    mlir::Type i1Ty = builder.getI1Type();
    yieldValue = builder.createIntegerConstant(currentLocation, i1Ty, 0);
    retTy.push_back(i1Ty);
  }

  auto loopOp = createRegionOp<mlir::acc::LoopOp, mlir::acc::YieldOp>(
      builder, builder.getFusedLoc(locs), currentLocation, eval, operands,
      operandSegments, /*outerCombined=*/false, retTy, yieldValue, ivTypes,
      ivLocs);

  for (auto [arg, value] : llvm::zip(
           loopOp.getLoopRegions().front()->front().getArguments(), ivPrivate))
    builder.create<fir::StoreOp>(currentLocation, arg, value);

  loopOp.setInclusiveUpperbound(inclusiveBounds);

  if (!gangDeviceTypes.empty())
    loopOp.setGangAttr(builder.getArrayAttr(gangDeviceTypes));
  if (!gangArgTypes.empty())
    loopOp.setGangOperandsArgTypeAttr(builder.getArrayAttr(gangArgTypes));
  if (!gangOperandsSegments.empty())
    loopOp.setGangOperandsSegmentsAttr(
        builder.getDenseI32ArrayAttr(gangOperandsSegments));
  if (!gangOperandsDeviceTypes.empty())
    loopOp.setGangOperandsDeviceTypeAttr(
        builder.getArrayAttr(gangOperandsDeviceTypes));

  if (!workerNumDeviceTypes.empty())
    loopOp.setWorkerAttr(builder.getArrayAttr(workerNumDeviceTypes));
  if (!workerNumOperandsDeviceTypes.empty())
    loopOp.setWorkerNumOperandsDeviceTypeAttr(
        builder.getArrayAttr(workerNumOperandsDeviceTypes));

  if (!vectorDeviceTypes.empty())
    loopOp.setVectorAttr(builder.getArrayAttr(vectorDeviceTypes));
  if (!vectorOperandsDeviceTypes.empty())
    loopOp.setVectorOperandsDeviceTypeAttr(
        builder.getArrayAttr(vectorOperandsDeviceTypes));

  if (!tileOperandsDeviceTypes.empty())
    loopOp.setTileOperandsDeviceTypeAttr(
        builder.getArrayAttr(tileOperandsDeviceTypes));
  if (!tileOperandsSegments.empty())
    loopOp.setTileOperandsSegmentsAttr(
        builder.getDenseI32ArrayAttr(tileOperandsSegments));

  if (!seqDeviceTypes.empty())
    loopOp.setSeqAttr(builder.getArrayAttr(seqDeviceTypes));
  if (!independentDeviceTypes.empty())
    loopOp.setIndependentAttr(builder.getArrayAttr(independentDeviceTypes));
  if (!autoDeviceTypes.empty())
    loopOp.setAuto_Attr(builder.getArrayAttr(autoDeviceTypes));

  if (!privatizations.empty())
    loopOp.setPrivatizationsAttr(
        mlir::ArrayAttr::get(builder.getContext(), privatizations));

  if (!reductionRecipes.empty())
    loopOp.setReductionRecipesAttr(
        mlir::ArrayAttr::get(builder.getContext(), reductionRecipes));

  if (!collapseValues.empty())
    loopOp.setCollapseAttr(builder.getI64ArrayAttr(collapseValues));
  if (!collapseDeviceTypes.empty())
    loopOp.setCollapseDeviceTypeAttr(builder.getArrayAttr(collapseDeviceTypes));

  return loopOp;
}

static bool hasEarlyReturn(Fortran::lower::pft::Evaluation &eval) {
  bool hasReturnStmt = false;
  for (auto &e : eval.getNestedEvaluations()) {
    e.visit(Fortran::common::visitors{
        [&](const Fortran::parser::ReturnStmt &) { hasReturnStmt = true; },
        [&](const auto &s) {},
    });
    if (e.hasNestedEvaluations())
      hasReturnStmt = hasEarlyReturn(e);
  }
  return hasReturnStmt;
}

static mlir::Value
genACC(Fortran::lower::AbstractConverter &converter,
       Fortran::semantics::SemanticsContext &semanticsContext,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenACCLoopConstruct &loopConstruct) {

  const auto &beginLoopDirective =
      std::get<Fortran::parser::AccBeginLoopDirective>(loopConstruct.t);
  const auto &loopDirective =
      std::get<Fortran::parser::AccLoopDirective>(beginLoopDirective.t);

  bool needEarlyExitHandling = false;
  if (eval.lowerAsUnstructured())
    needEarlyExitHandling = hasEarlyReturn(eval);

  mlir::Location currentLocation =
      converter.genLocation(beginLoopDirective.source);
  Fortran::lower::StatementContext stmtCtx;

  assert(loopDirective.v == llvm::acc::ACCD_loop &&
         "Unsupported OpenACC loop construct");
  (void)loopDirective;

  const auto &accClauseList =
      std::get<Fortran::parser::AccClauseList>(beginLoopDirective.t);
  const auto &outerDoConstruct =
      std::get<std::optional<Fortran::parser::DoConstruct>>(loopConstruct.t);
  auto loopOp = createLoopOp(converter, currentLocation, semanticsContext,
                             stmtCtx, *outerDoConstruct, eval, accClauseList,
                             needEarlyExitHandling);
  if (needEarlyExitHandling)
    return loopOp.getResult(0);

  return mlir::Value{};
}

template <typename Op, typename Clause>
static void genDataOperandOperationsWithModifier(
    const Clause *x, Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semanticsContext,
    Fortran::lower::StatementContext &stmtCtx,
    Fortran::parser::AccDataModifier::Modifier mod,
    llvm::SmallVectorImpl<mlir::Value> &dataClauseOperands,
    const mlir::acc::DataClause clause,
    const mlir::acc::DataClause clauseWithModifier,
    bool setDeclareAttr = false) {
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
                               /*structured=*/true, /*implicit=*/false,
                               setDeclareAttr);
}

template <typename Op>
static Op
createComputeOp(Fortran::lower::AbstractConverter &converter,
                mlir::Location currentLocation,
                Fortran::lower::pft::Evaluation &eval,
                Fortran::semantics::SemanticsContext &semanticsContext,
                Fortran::lower::StatementContext &stmtCtx,
                const Fortran::parser::AccClauseList &accClauseList,
                bool outerCombined = false) {

  // Parallel operation operands
  mlir::Value ifCond;
  mlir::Value selfCond;
  llvm::SmallVector<mlir::Value> waitOperands, attachEntryOperands,
      copyEntryOperands, copyoutEntryOperands, createEntryOperands,
      dataClauseOperands, numGangs, numWorkers, vectorLength, async;
  llvm::SmallVector<mlir::Attribute> numGangsDeviceTypes, numWorkersDeviceTypes,
      vectorLengthDeviceTypes, asyncDeviceTypes, asyncOnlyDeviceTypes,
      waitOperandsDeviceTypes, waitOnlyDeviceTypes;
  llvm::SmallVector<int32_t> numGangsSegments, waitOperandsSegments;

  llvm::SmallVector<mlir::Value> reductionOperands, privateOperands,
      firstprivateOperands;
  llvm::SmallVector<mlir::Attribute> privatizations, firstPrivatizations,
      reductionRecipes;
  mlir::Value waitDevnum; // TODO not yet implemented on compute op.

  // Self clause has optional values but can be present with
  // no value as well. When there is no value, the op has an attribute to
  // represent the clause.
  bool addSelfAttr = false;

  bool hasDefaultNone = false;
  bool hasDefaultPresent = false;

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  // device_type attribute is set to `none` until a device_type clause is
  // encountered.
  llvm::SmallVector<mlir::Attribute> crtDeviceTypes;
  auto crtDeviceTypeAttr = mlir::acc::DeviceTypeAttr::get(
      builder.getContext(), mlir::acc::DeviceType::None);
  crtDeviceTypes.push_back(crtDeviceTypeAttr);

  // Lower clauses values mapped to operands and array attributes.
  // Keep track of each group of operands separately as clauses can appear
  // more than once.
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *asyncClause =
            std::get_if<Fortran::parser::AccClause::Async>(&clause.u)) {
      genAsyncClause(converter, asyncClause, async, asyncDeviceTypes,
                     asyncOnlyDeviceTypes, crtDeviceTypes, stmtCtx);
    } else if (const auto *waitClause =
                   std::get_if<Fortran::parser::AccClause::Wait>(&clause.u)) {
      genWaitClause(converter, waitClause, waitOperands,
                    waitOperandsDeviceTypes, waitOnlyDeviceTypes,
                    waitOperandsSegments, waitDevnum, crtDeviceTypes, stmtCtx);
    } else if (const auto *numGangsClause =
                   std::get_if<Fortran::parser::AccClause::NumGangs>(
                       &clause.u)) {
      llvm::SmallVector<mlir::Value> numGangValues;
      for (const Fortran::parser::ScalarIntExpr &expr : numGangsClause->v)
        numGangValues.push_back(fir::getBase(converter.genExprValue(
            *Fortran::semantics::GetExpr(expr), stmtCtx)));
      for (auto crtDeviceTypeAttr : crtDeviceTypes) {
        for (auto value : numGangValues)
          numGangs.push_back(value);
        numGangsDeviceTypes.push_back(crtDeviceTypeAttr);
        numGangsSegments.push_back(numGangValues.size());
      }
    } else if (const auto *numWorkersClause =
                   std::get_if<Fortran::parser::AccClause::NumWorkers>(
                       &clause.u)) {
      mlir::Value numWorkerValue = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(numWorkersClause->v), stmtCtx));
      for (auto crtDeviceTypeAttr : crtDeviceTypes) {
        numWorkers.push_back(numWorkerValue);
        numWorkersDeviceTypes.push_back(crtDeviceTypeAttr);
      }
    } else if (const auto *vectorLengthClause =
                   std::get_if<Fortran::parser::AccClause::VectorLength>(
                       &clause.u)) {
      mlir::Value vectorLengthValue = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(vectorLengthClause->v), stmtCtx));
      for (auto crtDeviceTypeAttr : crtDeviceTypes) {
        vectorLength.push_back(vectorLengthValue);
        vectorLengthDeviceTypes.push_back(crtDeviceTypeAttr);
      }
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
          /*structured=*/true, /*implicit=*/false);
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
          /*structured=*/true, /*implicit=*/false);
    } else if (const auto *presentClause =
                   std::get_if<Fortran::parser::AccClause::Present>(
                       &clause.u)) {
      genDataOperandOperations<mlir::acc::PresentOp>(
          presentClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_present,
          /*structured=*/true, /*implicit=*/false);
    } else if (const auto *devicePtrClause =
                   std::get_if<Fortran::parser::AccClause::Deviceptr>(
                       &clause.u)) {
      genDataOperandOperations<mlir::acc::DevicePtrOp>(
          devicePtrClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_deviceptr,
          /*structured=*/true, /*implicit=*/false);
    } else if (const auto *attachClause =
                   std::get_if<Fortran::parser::AccClause::Attach>(&clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperations<mlir::acc::AttachOp>(
          attachClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_attach,
          /*structured=*/true, /*implicit=*/false);
      attachEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                 dataClauseOperands.end());
    } else if (const auto *privateClause =
                   std::get_if<Fortran::parser::AccClause::Private>(
                       &clause.u)) {
      if (!outerCombined)
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
      // A reduction clause on a combined construct is treated as if it appeared
      // on the loop construct. So don't generate a reduction clause when it is
      // combined - delay it to the loop. However, a reduction clause on a
      // combined construct implies a copy clause so issue an implicit copy
      // instead.
      if (!outerCombined) {
        genReductions(reductionClause->v, converter, semanticsContext, stmtCtx,
                      reductionOperands, reductionRecipes);
      } else {
        auto crtDataStart = dataClauseOperands.size();
        genDataOperandOperations<mlir::acc::CopyinOp>(
            std::get<Fortran::parser::AccObjectList>(reductionClause->v.t),
            converter, semanticsContext, stmtCtx, dataClauseOperands,
            mlir::acc::DataClause::acc_reduction,
            /*structured=*/true, /*implicit=*/true);
        copyEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                 dataClauseOperands.end());
      }
    } else if (const auto *defaultClause =
                   std::get_if<Fortran::parser::AccClause::Default>(
                       &clause.u)) {
      if ((defaultClause->v).v == llvm::acc::DefaultValue::ACC_Default_none)
        hasDefaultNone = true;
      else if ((defaultClause->v).v ==
               llvm::acc::DefaultValue::ACC_Default_present)
        hasDefaultPresent = true;
    } else if (const auto *deviceTypeClause =
                   std::get_if<Fortran::parser::AccClause::DeviceType>(
                       &clause.u)) {
      crtDeviceTypes.clear();
      gatherDeviceTypeAttrs(builder, deviceTypeClause, crtDeviceTypes);
    }
  }

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value, 8> operands;
  llvm::SmallVector<int32_t, 8> operandSegments;
  addOperands(operands, operandSegments, async);
  addOperands(operands, operandSegments, waitOperands);
  if constexpr (!std::is_same_v<Op, mlir::acc::SerialOp>) {
    addOperands(operands, operandSegments, numGangs);
    addOperands(operands, operandSegments, numWorkers);
    addOperands(operands, operandSegments, vectorLength);
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
        builder, currentLocation, currentLocation, eval, operands,
        operandSegments, outerCombined);
  else
    computeOp = createRegionOp<Op, mlir::acc::YieldOp>(
        builder, currentLocation, currentLocation, eval, operands,
        operandSegments, outerCombined);

  if (addSelfAttr)
    computeOp.setSelfAttrAttr(builder.getUnitAttr());

  if (hasDefaultNone)
    computeOp.setDefaultAttr(mlir::acc::ClauseDefaultValue::None);
  if (hasDefaultPresent)
    computeOp.setDefaultAttr(mlir::acc::ClauseDefaultValue::Present);

  if constexpr (!std::is_same_v<Op, mlir::acc::SerialOp>) {
    if (!numWorkersDeviceTypes.empty())
      computeOp.setNumWorkersDeviceTypeAttr(
          mlir::ArrayAttr::get(builder.getContext(), numWorkersDeviceTypes));
    if (!vectorLengthDeviceTypes.empty())
      computeOp.setVectorLengthDeviceTypeAttr(
          mlir::ArrayAttr::get(builder.getContext(), vectorLengthDeviceTypes));
    if (!numGangsDeviceTypes.empty())
      computeOp.setNumGangsDeviceTypeAttr(
          mlir::ArrayAttr::get(builder.getContext(), numGangsDeviceTypes));
    if (!numGangsSegments.empty())
      computeOp.setNumGangsSegmentsAttr(
          builder.getDenseI32ArrayAttr(numGangsSegments));
  }
  if (!asyncDeviceTypes.empty())
    computeOp.setAsyncDeviceTypeAttr(builder.getArrayAttr(asyncDeviceTypes));
  if (!asyncOnlyDeviceTypes.empty())
    computeOp.setAsyncOnlyAttr(builder.getArrayAttr(asyncOnlyDeviceTypes));

  if (!waitOperandsDeviceTypes.empty())
    computeOp.setWaitOperandsDeviceTypeAttr(
        builder.getArrayAttr(waitOperandsDeviceTypes));
  if (!waitOperandsSegments.empty())
    computeOp.setWaitOperandsSegmentsAttr(
        builder.getDenseI32ArrayAttr(waitOperandsSegments));
  if (!waitOnlyDeviceTypes.empty())
    computeOp.setWaitOnlyAttr(builder.getArrayAttr(waitOnlyDeviceTypes));

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
      builder, copyEntryOperands, /*structured=*/true);
  genDataExitOperations<mlir::acc::CreateOp, mlir::acc::CopyoutOp>(
      builder, copyoutEntryOperands, /*structured=*/true);
  genDataExitOperations<mlir::acc::AttachOp, mlir::acc::DetachOp>(
      builder, attachEntryOperands, /*structured=*/true);
  genDataExitOperations<mlir::acc::CreateOp, mlir::acc::DeleteOp>(
      builder, createEntryOperands, /*structured=*/true);

  builder.restoreInsertionPoint(insPt);
  return computeOp;
}

static void genACCDataOp(Fortran::lower::AbstractConverter &converter,
                         mlir::Location currentLocation,
                         Fortran::lower::pft::Evaluation &eval,
                         Fortran::semantics::SemanticsContext &semanticsContext,
                         Fortran::lower::StatementContext &stmtCtx,
                         const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value ifCond, waitDevnum;
  llvm::SmallVector<mlir::Value> attachEntryOperands, createEntryOperands,
      copyEntryOperands, copyoutEntryOperands, dataClauseOperands, waitOperands,
      async;
  llvm::SmallVector<mlir::Attribute> asyncDeviceTypes, asyncOnlyDeviceTypes,
      waitOperandsDeviceTypes, waitOnlyDeviceTypes;
  llvm::SmallVector<int32_t> waitOperandsSegments;

  bool hasDefaultNone = false;
  bool hasDefaultPresent = false;

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  // device_type attribute is set to `none` until a device_type clause is
  // encountered.
  llvm::SmallVector<mlir::Attribute> crtDeviceTypes;
  crtDeviceTypes.push_back(mlir::acc::DeviceTypeAttr::get(
      builder.getContext(), mlir::acc::DeviceType::None));

  // Lower clauses values mapped to operands and array attributes.
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
          /*structured=*/true, /*implicit=*/false);
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
          /*structured=*/true, /*implicit=*/false);
    } else if (const auto *presentClause =
                   std::get_if<Fortran::parser::AccClause::Present>(
                       &clause.u)) {
      genDataOperandOperations<mlir::acc::PresentOp>(
          presentClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_present,
          /*structured=*/true, /*implicit=*/false);
    } else if (const auto *deviceptrClause =
                   std::get_if<Fortran::parser::AccClause::Deviceptr>(
                       &clause.u)) {
      genDataOperandOperations<mlir::acc::DevicePtrOp>(
          deviceptrClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_deviceptr,
          /*structured=*/true, /*implicit=*/false);
    } else if (const auto *attachClause =
                   std::get_if<Fortran::parser::AccClause::Attach>(&clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDataOperandOperations<mlir::acc::AttachOp>(
          attachClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_attach,
          /*structured=*/true, /*implicit=*/false);
      attachEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                 dataClauseOperands.end());
    } else if (const auto *asyncClause =
                   std::get_if<Fortran::parser::AccClause::Async>(&clause.u)) {
      genAsyncClause(converter, asyncClause, async, asyncDeviceTypes,
                     asyncOnlyDeviceTypes, crtDeviceTypes, stmtCtx);
    } else if (const auto *waitClause =
                   std::get_if<Fortran::parser::AccClause::Wait>(&clause.u)) {
      genWaitClause(converter, waitClause, waitOperands,
                    waitOperandsDeviceTypes, waitOnlyDeviceTypes,
                    waitOperandsSegments, waitDevnum, crtDeviceTypes, stmtCtx);
    } else if(const auto *defaultClause = 
                  std::get_if<Fortran::parser::AccClause::Default>(&clause.u)) {
      if ((defaultClause->v).v == llvm::acc::DefaultValue::ACC_Default_none)
        hasDefaultNone = true;
      else if ((defaultClause->v).v == llvm::acc::DefaultValue::ACC_Default_present)
        hasDefaultPresent = true;
    } else if (const auto *deviceTypeClause =
                   std::get_if<Fortran::parser::AccClause::DeviceType>(
                       &clause.u)) {
      crtDeviceTypes.clear();
      gatherDeviceTypeAttrs(builder, deviceTypeClause, crtDeviceTypes);
    }
  }

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<int32_t> operandSegments;
  addOperand(operands, operandSegments, ifCond);
  addOperands(operands, operandSegments, async);
  addOperand(operands, operandSegments, waitDevnum);
  addOperands(operands, operandSegments, waitOperands);
  addOperands(operands, operandSegments, dataClauseOperands);

  if (dataClauseOperands.empty() && !hasDefaultNone && !hasDefaultPresent)
    return;

  auto dataOp = createRegionOp<mlir::acc::DataOp, mlir::acc::TerminatorOp>(
      builder, currentLocation, currentLocation, eval, operands,
      operandSegments);

  if (!asyncDeviceTypes.empty())
    dataOp.setAsyncDeviceTypeAttr(builder.getArrayAttr(asyncDeviceTypes));
  if (!asyncOnlyDeviceTypes.empty())
    dataOp.setAsyncOnlyAttr(builder.getArrayAttr(asyncOnlyDeviceTypes));
  if (!waitOperandsDeviceTypes.empty())
    dataOp.setWaitOperandsDeviceTypeAttr(
        builder.getArrayAttr(waitOperandsDeviceTypes));
  if (!waitOperandsSegments.empty())
    dataOp.setWaitOperandsSegmentsAttr(
        builder.getDenseI32ArrayAttr(waitOperandsSegments));
  if (!waitOnlyDeviceTypes.empty())
    dataOp.setWaitOnlyAttr(builder.getArrayAttr(waitOnlyDeviceTypes));

  if (hasDefaultNone)
    dataOp.setDefaultAttr(mlir::acc::ClauseDefaultValue::None);
  if (hasDefaultPresent)
    dataOp.setDefaultAttr(mlir::acc::ClauseDefaultValue::Present);

  auto insPt = builder.saveInsertionPoint();
  builder.setInsertionPointAfter(dataOp);

  // Create the exit operations after the region.
  genDataExitOperations<mlir::acc::CopyinOp, mlir::acc::CopyoutOp>(
      builder, copyEntryOperands, /*structured=*/true);
  genDataExitOperations<mlir::acc::CreateOp, mlir::acc::CopyoutOp>(
      builder, copyoutEntryOperands, /*structured=*/true);
  genDataExitOperations<mlir::acc::AttachOp, mlir::acc::DetachOp>(
      builder, attachEntryOperands, /*structured=*/true);
  genDataExitOperations<mlir::acc::CreateOp, mlir::acc::DeleteOp>(
      builder, createEntryOperands, /*structured=*/true);

  builder.restoreInsertionPoint(insPt);
}

static void
genACCHostDataOp(Fortran::lower::AbstractConverter &converter,
                 mlir::Location currentLocation,
                 Fortran::lower::pft::Evaluation &eval,
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
          /*structured=*/true, /*implicit=*/false);
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
          builder, currentLocation, currentLocation, eval, operands,
          operandSegments);

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
    createComputeOp<mlir::acc::ParallelOp>(converter, currentLocation, eval,
                                           semanticsContext, stmtCtx,
                                           accClauseList);
  } else if (blockDirective.v == llvm::acc::ACCD_data) {
    genACCDataOp(converter, currentLocation, eval, semanticsContext, stmtCtx,
                 accClauseList);
  } else if (blockDirective.v == llvm::acc::ACCD_serial) {
    createComputeOp<mlir::acc::SerialOp>(converter, currentLocation, eval,
                                         semanticsContext, stmtCtx,
                                         accClauseList);
  } else if (blockDirective.v == llvm::acc::ACCD_kernels) {
    createComputeOp<mlir::acc::KernelsOp>(converter, currentLocation, eval,
                                          semanticsContext, stmtCtx,
                                          accClauseList);
  } else if (blockDirective.v == llvm::acc::ACCD_host_data) {
    genACCHostDataOp(converter, currentLocation, eval, semanticsContext,
                     stmtCtx, accClauseList);
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
  const auto &outerDoConstruct =
      std::get<std::optional<Fortran::parser::DoConstruct>>(
          combinedConstruct.t);

  mlir::Location currentLocation =
      converter.genLocation(beginCombinedDirective.source);
  Fortran::lower::StatementContext stmtCtx;

  if (combinedDirective.v == llvm::acc::ACCD_kernels_loop) {
    createComputeOp<mlir::acc::KernelsOp>(
        converter, currentLocation, eval, semanticsContext, stmtCtx,
        accClauseList, /*outerCombined=*/true);
    createLoopOp(converter, currentLocation, semanticsContext, stmtCtx,
                 *outerDoConstruct, eval, accClauseList);
  } else if (combinedDirective.v == llvm::acc::ACCD_parallel_loop) {
    createComputeOp<mlir::acc::ParallelOp>(
        converter, currentLocation, eval, semanticsContext, stmtCtx,
        accClauseList, /*outerCombined=*/true);
    createLoopOp(converter, currentLocation, semanticsContext, stmtCtx,
                 *outerDoConstruct, eval, accClauseList);
  } else if (combinedDirective.v == llvm::acc::ACCD_serial_loop) {
    createComputeOp<mlir::acc::SerialOp>(converter, currentLocation, eval,
                                         semanticsContext, stmtCtx,
                                         accClauseList, /*outerCombined=*/true);
    createLoopOp(converter, currentLocation, semanticsContext, stmtCtx,
                 *outerDoConstruct, eval, accClauseList);
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
          dataClauseOperands, mlir::acc::DataClause::acc_copyin, false,
          /*implicit=*/false);
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
          dataClauseOperands, clause, false, /*implicit=*/false);
    } else if (const auto *attachClause =
                   std::get_if<Fortran::parser::AccClause::Attach>(&clause.u)) {
      genDataOperandOperations<mlir::acc::AttachOp>(
          attachClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_attach, false,
          /*implicit=*/false);
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
          mlir::acc::DataClause::acc_copyout, false, /*implicit=*/false);
    } else if (const auto *deleteClause =
                   std::get_if<Fortran::parser::AccClause::Delete>(&clause.u)) {
      genDataOperandOperations<mlir::acc::GetDevicePtrOp>(
          deleteClause->v, converter, semanticsContext, stmtCtx, deleteOperands,
          mlir::acc::DataClause::acc_delete, false, /*implicit=*/false);
    } else if (const auto *detachClause =
                   std::get_if<Fortran::parser::AccClause::Detach>(&clause.u)) {
      genDataOperandOperations<mlir::acc::GetDevicePtrOp>(
          detachClause->v, converter, semanticsContext, stmtCtx, detachOperands,
          mlir::acc::DataClause::acc_detach, false, /*implicit=*/false);
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
      builder, copyoutOperands, /*structured=*/false);
  genDataExitOperations<mlir::acc::GetDevicePtrOp, mlir::acc::DeleteOp>(
      builder, deleteOperands, /*structured=*/false);
  genDataExitOperations<mlir::acc::GetDevicePtrOp, mlir::acc::DetachOp>(
      builder, detachOperands, /*structured=*/false);
}

template <typename Op>
static void
genACCInitShutdownOp(Fortran::lower::AbstractConverter &converter,
                     mlir::Location currentLocation,
                     const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value ifCond, deviceNum;

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  llvm::SmallVector<mlir::Attribute> deviceTypes;

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
      gatherDeviceTypeAttrs(builder, deviceTypeClause, deviceTypes);
    }
  }

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value, 6> operands;
  llvm::SmallVector<int32_t, 2> operandSegments;

  addOperand(operands, operandSegments, deviceNum);
  addOperand(operands, operandSegments, ifCond);

  Op op =
      createSimpleOp<Op>(builder, currentLocation, operands, operandSegments);
  if (!deviceTypes.empty())
    op.setDeviceTypesAttr(
        mlir::ArrayAttr::get(builder.getContext(), deviceTypes));
}

void genACCSetOp(Fortran::lower::AbstractConverter &converter,
                 mlir::Location currentLocation,
                 const Fortran::parser::AccClauseList &accClauseList) {
  mlir::Value ifCond, deviceNum, defaultAsync;
  llvm::SmallVector<mlir::Value> deviceTypeOperands;

  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  llvm::SmallVector<mlir::Attribute> deviceTypes;

  // Lower clauses values mapped to operands.
  // Keep track of each group of operands separately as clauses can appear
  // more than once.
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (const auto *ifClause =
            std::get_if<Fortran::parser::AccClause::If>(&clause.u)) {
      genIfClause(converter, clauseLocation, ifClause, ifCond, stmtCtx);
    } else if (const auto *defaultAsyncClause =
                   std::get_if<Fortran::parser::AccClause::DefaultAsync>(
                       &clause.u)) {
      defaultAsync = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(defaultAsyncClause->v), stmtCtx));
    } else if (const auto *deviceNumClause =
                   std::get_if<Fortran::parser::AccClause::DeviceNum>(
                       &clause.u)) {
      deviceNum = fir::getBase(converter.genExprValue(
          *Fortran::semantics::GetExpr(deviceNumClause->v), stmtCtx));
    } else if (const auto *deviceTypeClause =
                   std::get_if<Fortran::parser::AccClause::DeviceType>(
                       &clause.u)) {
      gatherDeviceTypeAttrs(builder, deviceTypeClause, deviceTypes);
    }
  }

  // Prepare the operand segment size attribute and the operands value range.
  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<int32_t, 3> operandSegments;
  addOperand(operands, operandSegments, defaultAsync);
  addOperand(operands, operandSegments, deviceNum);
  addOperand(operands, operandSegments, ifCond);

  auto op = createSimpleOp<mlir::acc::SetOp>(builder, currentLocation, operands,
                                             operandSegments);
  if (!deviceTypes.empty()) {
    assert(deviceTypes.size() == 1 && "expect only one value for acc.set");
    op.setDeviceTypeAttr(mlir::cast<mlir::acc::DeviceTypeAttr>(deviceTypes[0]));
  }
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
  llvm::SmallVector<mlir::Attribute> deviceTypes;

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
      gatherDeviceTypeAttrs(builder, deviceTypeClause, deviceTypes);
    } else if (const auto *hostClause =
                   std::get_if<Fortran::parser::AccClause::Host>(&clause.u)) {
      genDataOperandOperations<mlir::acc::GetDevicePtrOp>(
          hostClause->v, converter, semanticsContext, stmtCtx,
          updateHostOperands, mlir::acc::DataClause::acc_update_host, false,
          /*implicit=*/false);
    } else if (const auto *deviceClause =
                   std::get_if<Fortran::parser::AccClause::Device>(&clause.u)) {
      genDataOperandOperations<mlir::acc::UpdateDeviceOp>(
          deviceClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_update_device, false,
          /*implicit=*/false);
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
          updateHostOperands, mlir::acc::DataClause::acc_update_self, false,
          /*implicit=*/false);
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
  addOperands(operands, operandSegments, dataClauseOperands);

  mlir::acc::UpdateOp updateOp = createSimpleOp<mlir::acc::UpdateOp>(
      builder, currentLocation, operands, operandSegments);
  if (!deviceTypes.empty())
    updateOp.setDeviceTypesAttr(
        mlir::ArrayAttr::get(builder.getContext(), deviceTypes));

  genDataExitOperations<mlir::acc::GetDevicePtrOp, mlir::acc::UpdateHostOp>(
      builder, updateHostOperands, /*structured=*/false);

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
    genACCSetOp(converter, currentLocation, accClauseList);
  } else if (standaloneDirective.v == llvm::acc::Directive::ACCD_update) {
    genACCUpdateOp(converter, currentLocation, semanticsContext, stmtCtx,
                   accClauseList);
  }
}

static void genACC(Fortran::lower::AbstractConverter &converter,
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

template <typename GlobalOp, typename EntryOp, typename DeclareOp,
          typename ExitOp>
static void createDeclareGlobalOp(mlir::OpBuilder &modBuilder,
                                  fir::FirOpBuilder &builder,
                                  mlir::Location loc, fir::GlobalOp globalOp,
                                  mlir::acc::DataClause clause,
                                  const std::string declareGlobalName,
                                  bool implicit, std::stringstream &asFortran) {
  GlobalOp declareGlobalOp =
      modBuilder.create<GlobalOp>(loc, declareGlobalName);
  builder.createBlock(&declareGlobalOp.getRegion(),
                      declareGlobalOp.getRegion().end(), {}, {});
  builder.setInsertionPointToEnd(&declareGlobalOp.getRegion().back());

  fir::AddrOfOp addrOp = builder.create<fir::AddrOfOp>(
      loc, fir::ReferenceType::get(globalOp.getType()), globalOp.getSymbol());
  addDeclareAttr(builder, addrOp, clause);

  llvm::SmallVector<mlir::Value> bounds;
  EntryOp entryOp = createDataEntryOp<EntryOp>(
      builder, loc, addrOp.getResTy(), asFortran, bounds,
      /*structured=*/false, implicit, clause, addrOp.getResTy().getType());
  if constexpr (std::is_same_v<DeclareOp, mlir::acc::DeclareEnterOp>)
    builder.create<DeclareOp>(
        loc, mlir::acc::DeclareTokenType::get(entryOp.getContext()),
        mlir::ValueRange(entryOp.getAccPtr()));
  else
    builder.create<DeclareOp>(loc, mlir::Value{},
                              mlir::ValueRange(entryOp.getAccPtr()));
  if constexpr (std::is_same_v<GlobalOp, mlir::acc::GlobalDestructorOp>) {
    builder.create<ExitOp>(entryOp.getLoc(), entryOp.getAccPtr(),
                           entryOp.getBounds(), entryOp.getDataClause(),
                           /*structured=*/false, /*implicit=*/false,
                           builder.getStringAttr(*entryOp.getName()));
  }
  builder.create<mlir::acc::TerminatorOp>(loc);
  modBuilder.setInsertionPointAfter(declareGlobalOp);
}

template <typename EntryOp>
static void createDeclareAllocFunc(mlir::OpBuilder &modBuilder,
                                   fir::FirOpBuilder &builder,
                                   mlir::Location loc, fir::GlobalOp &globalOp,
                                   mlir::acc::DataClause clause) {
  std::stringstream registerFuncName;
  registerFuncName << globalOp.getSymName().str()
                   << Fortran::lower::declarePostAllocSuffix.str();
  auto registerFuncOp =
      createDeclareFunc(modBuilder, builder, loc, registerFuncName.str());

  fir::AddrOfOp addrOp = builder.create<fir::AddrOfOp>(
      loc, fir::ReferenceType::get(globalOp.getType()), globalOp.getSymbol());

  std::stringstream asFortran;
  asFortran << Fortran::lower::mangle::demangleName(globalOp.getSymName());
  std::stringstream asFortranDesc;
  asFortranDesc << asFortran.str() << accFirDescriptorPostfix.str();
  llvm::SmallVector<mlir::Value> bounds;

  // Updating descriptor must occur before the mapping of the data so that
  // attached data pointer is not overwritten.
  mlir::acc::UpdateDeviceOp updateDeviceOp =
      createDataEntryOp<mlir::acc::UpdateDeviceOp>(
          builder, loc, addrOp, asFortranDesc, bounds,
          /*structured=*/false, /*implicit=*/true,
          mlir::acc::DataClause::acc_update_device, addrOp.getType());
  llvm::SmallVector<int32_t> operandSegments{0, 0, 0, 0, 1};
  llvm::SmallVector<mlir::Value> operands{updateDeviceOp.getResult()};
  createSimpleOp<mlir::acc::UpdateOp>(builder, loc, operands, operandSegments);

  auto loadOp = builder.create<fir::LoadOp>(loc, addrOp.getResult());
  fir::BoxAddrOp boxAddrOp = builder.create<fir::BoxAddrOp>(loc, loadOp);
  addDeclareAttr(builder, boxAddrOp.getOperation(), clause);
  EntryOp entryOp = createDataEntryOp<EntryOp>(
      builder, loc, boxAddrOp.getResult(), asFortran, bounds,
      /*structured=*/false, /*implicit=*/false, clause, boxAddrOp.getType());
  builder.create<mlir::acc::DeclareEnterOp>(
      loc, mlir::acc::DeclareTokenType::get(entryOp.getContext()),
      mlir::ValueRange(entryOp.getAccPtr()));

  modBuilder.setInsertionPointAfter(registerFuncOp);
}

/// Action to be performed on deallocation are split in two distinct functions.
/// - Pre deallocation function includes all the action to be performed before
///   the actual deallocation is done on the host side.
/// - Post deallocation function includes update to the descriptor.
template <typename ExitOp>
static void createDeclareDeallocFunc(mlir::OpBuilder &modBuilder,
                                     fir::FirOpBuilder &builder,
                                     mlir::Location loc,
                                     fir::GlobalOp &globalOp,
                                     mlir::acc::DataClause clause) {

  // Generate the pre dealloc function.
  std::stringstream preDeallocFuncName;
  preDeallocFuncName << globalOp.getSymName().str()
                     << Fortran::lower::declarePreDeallocSuffix.str();
  auto preDeallocOp =
      createDeclareFunc(modBuilder, builder, loc, preDeallocFuncName.str());
  fir::AddrOfOp addrOp = builder.create<fir::AddrOfOp>(
      loc, fir::ReferenceType::get(globalOp.getType()), globalOp.getSymbol());
  auto loadOp = builder.create<fir::LoadOp>(loc, addrOp.getResult());
  fir::BoxAddrOp boxAddrOp = builder.create<fir::BoxAddrOp>(loc, loadOp);
  addDeclareAttr(builder, boxAddrOp.getOperation(), clause);

  std::stringstream asFortran;
  asFortran << Fortran::lower::mangle::demangleName(globalOp.getSymName());
  llvm::SmallVector<mlir::Value> bounds;
  mlir::acc::GetDevicePtrOp entryOp =
      createDataEntryOp<mlir::acc::GetDevicePtrOp>(
          builder, loc, boxAddrOp.getResult(), asFortran, bounds,
          /*structured=*/false, /*implicit=*/false, clause,
          boxAddrOp.getType());

  builder.create<mlir::acc::DeclareExitOp>(
      loc, mlir::Value{}, mlir::ValueRange(entryOp.getAccPtr()));

  if constexpr (std::is_same_v<ExitOp, mlir::acc::CopyoutOp> ||
                std::is_same_v<ExitOp, mlir::acc::UpdateHostOp>)
    builder.create<ExitOp>(entryOp.getLoc(), entryOp.getAccPtr(),
                           entryOp.getVarPtr(), entryOp.getBounds(),
                           entryOp.getDataClause(),
                           /*structured=*/false, /*implicit=*/false,
                           builder.getStringAttr(*entryOp.getName()));
  else
    builder.create<ExitOp>(entryOp.getLoc(), entryOp.getAccPtr(),
                           entryOp.getBounds(), entryOp.getDataClause(),
                           /*structured=*/false, /*implicit=*/false,
                           builder.getStringAttr(*entryOp.getName()));

  // Generate the post dealloc function.
  modBuilder.setInsertionPointAfter(preDeallocOp);
  std::stringstream postDeallocFuncName;
  postDeallocFuncName << globalOp.getSymName().str()
                      << Fortran::lower::declarePostDeallocSuffix.str();
  auto postDeallocOp =
      createDeclareFunc(modBuilder, builder, loc, postDeallocFuncName.str());

  addrOp = builder.create<fir::AddrOfOp>(
      loc, fir::ReferenceType::get(globalOp.getType()), globalOp.getSymbol());
  asFortran << accFirDescriptorPostfix.str();
  mlir::acc::UpdateDeviceOp updateDeviceOp =
      createDataEntryOp<mlir::acc::UpdateDeviceOp>(
          builder, loc, addrOp, asFortran, bounds,
          /*structured=*/false, /*implicit=*/true,
          mlir::acc::DataClause::acc_update_device, addrOp.getType());
  llvm::SmallVector<int32_t> operandSegments{0, 0, 0, 0, 1};
  llvm::SmallVector<mlir::Value> operands{updateDeviceOp.getResult()};
  createSimpleOp<mlir::acc::UpdateOp>(builder, loc, operands, operandSegments);
  modBuilder.setInsertionPointAfter(postDeallocOp);
}

template <typename EntryOp, typename ExitOp>
static void genGlobalCtors(Fortran::lower::AbstractConverter &converter,
                           mlir::OpBuilder &modBuilder,
                           const Fortran::parser::AccObjectList &accObjectList,
                           mlir::acc::DataClause clause) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  for (const auto &accObject : accObjectList.v) {
    mlir::Location operandLocation = genOperandLocation(converter, accObject);
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::Designator &designator) {
              if (const auto *name =
                      Fortran::semantics::getDesignatorNameIfDataRef(
                          designator)) {
                std::string globalName = converter.mangleName(*name->symbol);
                fir::GlobalOp globalOp = builder.getNamedGlobal(globalName);
                std::stringstream declareGlobalCtorName;
                declareGlobalCtorName << globalName << "_acc_ctor";
                std::stringstream declareGlobalDtorName;
                declareGlobalDtorName << globalName << "_acc_dtor";
                std::stringstream asFortran;
                asFortran << name->symbol->name().ToString();

                if (builder.getModule()
                        .lookupSymbol<mlir::acc::GlobalConstructorOp>(
                            declareGlobalCtorName.str()))
                  return;

                if (!globalOp) {
                  if (Fortran::semantics::FindEquivalenceSet(*name->symbol)) {
                    for (Fortran::semantics::EquivalenceObject eqObj :
                         *Fortran::semantics::FindEquivalenceSet(
                             *name->symbol)) {
                      std::string eqName = converter.mangleName(eqObj.symbol);
                      globalOp = builder.getNamedGlobal(eqName);
                      if (globalOp)
                        break;
                    }

                    if (!globalOp)
                      llvm::report_fatal_error(
                          "could not retrieve global symbol");
                  } else {
                    llvm::report_fatal_error(
                        "could not retrieve global symbol");
                  }
                }

                addDeclareAttr(builder, globalOp.getOperation(), clause);
                auto crtPos = builder.saveInsertionPoint();
                modBuilder.setInsertionPointAfter(globalOp);
                if (mlir::isa<fir::BaseBoxType>(
                        fir::unwrapRefType(globalOp.getType()))) {
                  createDeclareGlobalOp<mlir::acc::GlobalConstructorOp,
                                        mlir::acc::CopyinOp,
                                        mlir::acc::DeclareEnterOp, ExitOp>(
                      modBuilder, builder, operandLocation, globalOp, clause,
                      declareGlobalCtorName.str(), /*implicit=*/true,
                      asFortran);
                  createDeclareAllocFunc<EntryOp>(
                      modBuilder, builder, operandLocation, globalOp, clause);
                  if constexpr (!std::is_same_v<EntryOp, ExitOp>)
                    createDeclareDeallocFunc<ExitOp>(
                        modBuilder, builder, operandLocation, globalOp, clause);
                } else {
                  createDeclareGlobalOp<mlir::acc::GlobalConstructorOp, EntryOp,
                                        mlir::acc::DeclareEnterOp, ExitOp>(
                      modBuilder, builder, operandLocation, globalOp, clause,
                      declareGlobalCtorName.str(), /*implicit=*/false,
                      asFortran);
                }
                if constexpr (!std::is_same_v<EntryOp, ExitOp>) {
                  createDeclareGlobalOp<mlir::acc::GlobalDestructorOp,
                                        mlir::acc::GetDevicePtrOp,
                                        mlir::acc::DeclareExitOp, ExitOp>(
                      modBuilder, builder, operandLocation, globalOp, clause,
                      declareGlobalDtorName.str(), /*implicit=*/false,
                      asFortran);
                }
                builder.restoreInsertionPoint(crtPos);
              }
            },
            [&](const Fortran::parser::Name &name) {
              TODO(operandLocation, "OpenACC Global Ctor from parser::Name");
            }},
        accObject.u);
  }
}

template <typename Clause, typename EntryOp, typename ExitOp>
static void
genGlobalCtorsWithModifier(Fortran::lower::AbstractConverter &converter,
                           mlir::OpBuilder &modBuilder, const Clause *x,
                           Fortran::parser::AccDataModifier::Modifier mod,
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
  genGlobalCtors<EntryOp, ExitOp>(converter, modBuilder, accObjectList,
                                  dataClause);
}

static void
genDeclareInFunction(Fortran::lower::AbstractConverter &converter,
                     Fortran::semantics::SemanticsContext &semanticsContext,
                     Fortran::lower::StatementContext &openAccCtx,
                     mlir::Location loc,
                     const Fortran::parser::AccClauseList &accClauseList) {
  llvm::SmallVector<mlir::Value> dataClauseOperands, copyEntryOperands,
      createEntryOperands, copyoutEntryOperands, deviceResidentEntryOperands;
  Fortran::lower::StatementContext stmtCtx;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    if (const auto *copyClause =
            std::get_if<Fortran::parser::AccClause::Copy>(&clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDeclareDataOperandOperations<mlir::acc::CopyinOp,
                                      mlir::acc::CopyoutOp>(
          copyClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_copy,
          /*structured=*/true, /*implicit=*/false);
      copyEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                               dataClauseOperands.end());
    } else if (const auto *createClause =
                   std::get_if<Fortran::parser::AccClause::Create>(&clause.u)) {
      const Fortran::parser::AccObjectListWithModifier &listWithModifier =
          createClause->v;
      const auto &accObjectList =
          std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
      auto crtDataStart = dataClauseOperands.size();
      genDeclareDataOperandOperations<mlir::acc::CreateOp, mlir::acc::DeleteOp>(
          accObjectList, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_create,
          /*structured=*/true, /*implicit=*/false);
      createEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                 dataClauseOperands.end());
    } else if (const auto *presentClause =
                   std::get_if<Fortran::parser::AccClause::Present>(
                       &clause.u)) {
      genDeclareDataOperandOperations<mlir::acc::PresentOp,
                                      mlir::acc::PresentOp>(
          presentClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_present,
          /*structured=*/true, /*implicit=*/false);
    } else if (const auto *copyinClause =
                   std::get_if<Fortran::parser::AccClause::Copyin>(&clause.u)) {
      genDeclareDataOperandOperationsWithModifier<mlir::acc::CopyinOp,
                                                  mlir::acc::DeleteOp>(
          copyinClause, converter, semanticsContext, stmtCtx,
          Fortran::parser::AccDataModifier::Modifier::ReadOnly,
          dataClauseOperands, mlir::acc::DataClause::acc_copyin,
          mlir::acc::DataClause::acc_copyin_readonly);
    } else if (const auto *copyoutClause =
                   std::get_if<Fortran::parser::AccClause::Copyout>(
                       &clause.u)) {
      const Fortran::parser::AccObjectListWithModifier &listWithModifier =
          copyoutClause->v;
      const auto &accObjectList =
          std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
      auto crtDataStart = dataClauseOperands.size();
      genDeclareDataOperandOperations<mlir::acc::CreateOp,
                                      mlir::acc::CopyoutOp>(
          accObjectList, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_copyout,
          /*structured=*/true, /*implicit=*/false);
      copyoutEntryOperands.append(dataClauseOperands.begin() + crtDataStart,
                                  dataClauseOperands.end());
    } else if (const auto *devicePtrClause =
                   std::get_if<Fortran::parser::AccClause::Deviceptr>(
                       &clause.u)) {
      genDeclareDataOperandOperations<mlir::acc::DevicePtrOp,
                                      mlir::acc::DevicePtrOp>(
          devicePtrClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_deviceptr,
          /*structured=*/true, /*implicit=*/false);
    } else if (const auto *linkClause =
                   std::get_if<Fortran::parser::AccClause::Link>(&clause.u)) {
      genDeclareDataOperandOperations<mlir::acc::DeclareLinkOp,
                                      mlir::acc::DeclareLinkOp>(
          linkClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands, mlir::acc::DataClause::acc_declare_link,
          /*structured=*/true, /*implicit=*/false);
    } else if (const auto *deviceResidentClause =
                   std::get_if<Fortran::parser::AccClause::DeviceResident>(
                       &clause.u)) {
      auto crtDataStart = dataClauseOperands.size();
      genDeclareDataOperandOperations<mlir::acc::DeclareDeviceResidentOp,
                                      mlir::acc::DeleteOp>(
          deviceResidentClause->v, converter, semanticsContext, stmtCtx,
          dataClauseOperands,
          mlir::acc::DataClause::acc_declare_device_resident,
          /*structured=*/true, /*implicit=*/false);
      deviceResidentEntryOperands.append(
          dataClauseOperands.begin() + crtDataStart, dataClauseOperands.end());
    } else {
      mlir::Location clauseLocation = converter.genLocation(clause.source);
      TODO(clauseLocation, "clause on declare directive");
    }
  }

  mlir::func::FuncOp funcOp = builder.getFunction();
  auto ops = funcOp.getOps<mlir::acc::DeclareEnterOp>();
  mlir::Value declareToken;
  if (ops.empty()) {
    declareToken = builder.create<mlir::acc::DeclareEnterOp>(
        loc, mlir::acc::DeclareTokenType::get(builder.getContext()),
        dataClauseOperands);
  } else {
    auto declareOp = *ops.begin();
    auto newDeclareOp = builder.create<mlir::acc::DeclareEnterOp>(
        loc, mlir::acc::DeclareTokenType::get(builder.getContext()),
        declareOp.getDataClauseOperands());
    newDeclareOp.getDataClauseOperandsMutable().append(dataClauseOperands);
    declareToken = newDeclareOp.getToken();
    declareOp.erase();
  }

  openAccCtx.attachCleanup([&builder, loc, createEntryOperands,
                            copyEntryOperands, copyoutEntryOperands,
                            deviceResidentEntryOperands, declareToken]() {
    llvm::SmallVector<mlir::Value> operands;
    operands.append(createEntryOperands);
    operands.append(deviceResidentEntryOperands);
    operands.append(copyEntryOperands);
    operands.append(copyoutEntryOperands);

    mlir::func::FuncOp funcOp = builder.getFunction();
    auto ops = funcOp.getOps<mlir::acc::DeclareExitOp>();
    if (ops.empty()) {
      builder.create<mlir::acc::DeclareExitOp>(loc, declareToken, operands);
    } else {
      auto declareOp = *ops.begin();
      declareOp.getDataClauseOperandsMutable().append(operands);
    }

    genDataExitOperations<mlir::acc::CreateOp, mlir::acc::DeleteOp>(
        builder, createEntryOperands, /*structured=*/true);
    genDataExitOperations<mlir::acc::DeclareDeviceResidentOp,
                          mlir::acc::DeleteOp>(
        builder, deviceResidentEntryOperands, /*structured=*/true);
    genDataExitOperations<mlir::acc::CopyinOp, mlir::acc::CopyoutOp>(
        builder, copyEntryOperands, /*structured=*/true);
    genDataExitOperations<mlir::acc::CreateOp, mlir::acc::CopyoutOp>(
        builder, copyoutEntryOperands, /*structured=*/true);
  });
}

static void
genDeclareInModule(Fortran::lower::AbstractConverter &converter,
                   mlir::ModuleOp &moduleOp,
                   const Fortran::parser::AccClauseList &accClauseList) {
  mlir::OpBuilder modBuilder(moduleOp.getBodyRegion());
  for (const Fortran::parser::AccClause &clause : accClauseList.v) {
    if (const auto *createClause =
            std::get_if<Fortran::parser::AccClause::Create>(&clause.u)) {
      const Fortran::parser::AccObjectListWithModifier &listWithModifier =
          createClause->v;
      const auto &accObjectList =
          std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
      genGlobalCtors<mlir::acc::CreateOp, mlir::acc::DeleteOp>(
          converter, modBuilder, accObjectList,
          mlir::acc::DataClause::acc_create);
    } else if (const auto *copyinClause =
                   std::get_if<Fortran::parser::AccClause::Copyin>(&clause.u)) {
      genGlobalCtorsWithModifier<Fortran::parser::AccClause::Copyin,
                                 mlir::acc::CopyinOp, mlir::acc::CopyinOp>(
          converter, modBuilder, copyinClause,
          Fortran::parser::AccDataModifier::Modifier::ReadOnly,
          mlir::acc::DataClause::acc_copyin,
          mlir::acc::DataClause::acc_copyin_readonly);
    } else if (const auto *deviceResidentClause =
                   std::get_if<Fortran::parser::AccClause::DeviceResident>(
                       &clause.u)) {
      genGlobalCtors<mlir::acc::DeclareDeviceResidentOp, mlir::acc::DeleteOp>(
          converter, modBuilder, deviceResidentClause->v,
          mlir::acc::DataClause::acc_declare_device_resident);
    } else if (const auto *linkClause =
                   std::get_if<Fortran::parser::AccClause::Link>(&clause.u)) {
      genGlobalCtors<mlir::acc::DeclareLinkOp, mlir::acc::DeclareLinkOp>(
          converter, modBuilder, linkClause->v,
          mlir::acc::DataClause::acc_declare_link);
    } else {
      llvm::report_fatal_error("unsupported clause on DECLARE directive");
    }
  }
}

static void genACC(Fortran::lower::AbstractConverter &converter,
                   Fortran::semantics::SemanticsContext &semanticsContext,
                   Fortran::lower::StatementContext &openAccCtx,
                   const Fortran::parser::OpenACCStandaloneDeclarativeConstruct
                       &declareConstruct) {

  const auto &declarativeDir =
      std::get<Fortran::parser::AccDeclarativeDirective>(declareConstruct.t);
  mlir::Location directiveLocation =
      converter.genLocation(declarativeDir.source);
  const auto &accClauseList =
      std::get<Fortran::parser::AccClauseList>(declareConstruct.t);

  if (declarativeDir.v == llvm::acc::Directive::ACCD_declare) {
    fir::FirOpBuilder &builder = converter.getFirOpBuilder();
    auto moduleOp =
        builder.getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();
    auto funcOp =
        builder.getBlock()->getParent()->getParentOfType<mlir::func::FuncOp>();
    if (funcOp)
      genDeclareInFunction(converter, semanticsContext, openAccCtx,
                           directiveLocation, accClauseList);
    else if (moduleOp)
      genDeclareInModule(converter, moduleOp, accClauseList);
    return;
  }
  llvm_unreachable("unsupported declarative directive");
}

static bool hasDeviceType(llvm::SmallVector<mlir::Attribute> &arrayAttr,
                          mlir::acc::DeviceType deviceType) {
  for (auto attr : arrayAttr) {
    auto deviceTypeAttr = mlir::dyn_cast<mlir::acc::DeviceTypeAttr>(attr);
    if (deviceTypeAttr.getValue() == deviceType)
      return true;
  }
  return false;
}

template <typename RetTy, typename AttrTy>
static std::optional<RetTy>
getAttributeValueByDeviceType(llvm::SmallVector<mlir::Attribute> &attributes,
                              llvm::SmallVector<mlir::Attribute> &deviceTypes,
                              mlir::acc::DeviceType deviceType) {
  assert(attributes.size() == deviceTypes.size() &&
         "expect same number of attributes");
  for (auto it : llvm::enumerate(deviceTypes)) {
    auto deviceTypeAttr = mlir::dyn_cast<mlir::acc::DeviceTypeAttr>(it.value());
    if (deviceTypeAttr.getValue() == deviceType) {
      if constexpr (std::is_same_v<mlir::StringAttr, AttrTy>) {
        auto strAttr = mlir::dyn_cast<AttrTy>(attributes[it.index()]);
        return strAttr.getValue();
      } else if constexpr (std::is_same_v<mlir::IntegerAttr, AttrTy>) {
        auto intAttr =
            mlir::dyn_cast<mlir::IntegerAttr>(attributes[it.index()]);
        return intAttr.getInt();
      }
    }
  }
  return std::nullopt;
}

static bool compareDeviceTypeInfo(
    mlir::acc::RoutineOp op,
    llvm::SmallVector<mlir::Attribute> &bindNameArrayAttr,
    llvm::SmallVector<mlir::Attribute> &bindNameDeviceTypeArrayAttr,
    llvm::SmallVector<mlir::Attribute> &gangArrayAttr,
    llvm::SmallVector<mlir::Attribute> &gangDimArrayAttr,
    llvm::SmallVector<mlir::Attribute> &gangDimDeviceTypeArrayAttr,
    llvm::SmallVector<mlir::Attribute> &seqArrayAttr,
    llvm::SmallVector<mlir::Attribute> &workerArrayAttr,
    llvm::SmallVector<mlir::Attribute> &vectorArrayAttr) {
  for (uint32_t dtypeInt = 0;
       dtypeInt != mlir::acc::getMaxEnumValForDeviceType(); ++dtypeInt) {
    auto dtype = static_cast<mlir::acc::DeviceType>(dtypeInt);
    if (op.getBindNameValue(dtype) !=
        getAttributeValueByDeviceType<llvm::StringRef, mlir::StringAttr>(
            bindNameArrayAttr, bindNameDeviceTypeArrayAttr, dtype))
      return false;
    if (op.hasGang(dtype) != hasDeviceType(gangArrayAttr, dtype))
      return false;
    if (op.getGangDimValue(dtype) !=
        getAttributeValueByDeviceType<int64_t, mlir::IntegerAttr>(
            gangDimArrayAttr, gangDimDeviceTypeArrayAttr, dtype))
      return false;
    if (op.hasSeq(dtype) != hasDeviceType(seqArrayAttr, dtype))
      return false;
    if (op.hasWorker(dtype) != hasDeviceType(workerArrayAttr, dtype))
      return false;
    if (op.hasVector(dtype) != hasDeviceType(vectorArrayAttr, dtype))
      return false;
  }
  return true;
}

static void attachRoutineInfo(mlir::func::FuncOp func,
                              mlir::SymbolRefAttr routineAttr) {
  llvm::SmallVector<mlir::SymbolRefAttr> routines;
  if (func.getOperation()->hasAttr(mlir::acc::getRoutineInfoAttrName())) {
    auto routineInfo =
        func.getOperation()->getAttrOfType<mlir::acc::RoutineInfoAttr>(
            mlir::acc::getRoutineInfoAttrName());
    routines.append(routineInfo.getAccRoutines().begin(),
                    routineInfo.getAccRoutines().end());
  }
  routines.push_back(routineAttr);
  func.getOperation()->setAttr(
      mlir::acc::getRoutineInfoAttrName(),
      mlir::acc::RoutineInfoAttr::get(func.getContext(), routines));
}

void Fortran::lower::genOpenACCRoutineConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semanticsContext, mlir::ModuleOp &mod,
    const Fortran::parser::OpenACCRoutineConstruct &routineConstruct,
    Fortran::lower::AccRoutineInfoMappingList &accRoutineInfos) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.genLocation(routineConstruct.source);
  std::optional<Fortran::parser::Name> name =
      std::get<std::optional<Fortran::parser::Name>>(routineConstruct.t);
  const auto &clauses =
      std::get<Fortran::parser::AccClauseList>(routineConstruct.t);
  mlir::func::FuncOp funcOp;
  std::string funcName;
  if (name) {
    funcName = converter.mangleName(*name->symbol);
    funcOp = builder.getNamedFunction(mod, funcName);
  } else {
    Fortran::semantics::Scope &scope =
        semanticsContext.FindScope(routineConstruct.source);
    const Fortran::semantics::Scope &progUnit{GetProgramUnitContaining(scope)};
    const auto *subpDetails{
        progUnit.symbol()
            ? progUnit.symbol()
                  ->detailsIf<Fortran::semantics::SubprogramDetails>()
            : nullptr};
    if (subpDetails && subpDetails->isInterface()) {
      funcName = converter.mangleName(*progUnit.symbol());
      funcOp = builder.getNamedFunction(mod, funcName);
    } else {
      funcOp = builder.getFunction();
      funcName = funcOp.getName();
    }
  }
  bool hasNohost = false;

  llvm::SmallVector<mlir::Attribute> seqDeviceTypes, vectorDeviceTypes,
      workerDeviceTypes, bindNameDeviceTypes, bindNames, gangDeviceTypes,
      gangDimDeviceTypes, gangDimValues;

  // device_type attribute is set to `none` until a device_type clause is
  // encountered.
  llvm::SmallVector<mlir::Attribute> crtDeviceTypes;
  crtDeviceTypes.push_back(mlir::acc::DeviceTypeAttr::get(
      builder.getContext(), mlir::acc::DeviceType::None));

  for (const Fortran::parser::AccClause &clause : clauses.v) {
    if (std::get_if<Fortran::parser::AccClause::Seq>(&clause.u)) {
      for (auto crtDeviceTypeAttr : crtDeviceTypes)
        seqDeviceTypes.push_back(crtDeviceTypeAttr);
    } else if (const auto *gangClause =
                   std::get_if<Fortran::parser::AccClause::Gang>(&clause.u)) {
      if (gangClause->v) {
        const Fortran::parser::AccGangArgList &x = *gangClause->v;
        for (const Fortran::parser::AccGangArg &gangArg : x.v) {
          if (const auto *dim =
                  std::get_if<Fortran::parser::AccGangArg::Dim>(&gangArg.u)) {
            const std::optional<int64_t> dimValue = Fortran::evaluate::ToInt64(
                *Fortran::semantics::GetExpr(dim->v));
            if (!dimValue)
              mlir::emitError(loc,
                              "dim value must be a constant positive integer");
            mlir::Attribute gangDimAttr =
                builder.getIntegerAttr(builder.getI64Type(), *dimValue);
            for (auto crtDeviceTypeAttr : crtDeviceTypes) {
              gangDimValues.push_back(gangDimAttr);
              gangDimDeviceTypes.push_back(crtDeviceTypeAttr);
            }
          }
        }
      } else {
        for (auto crtDeviceTypeAttr : crtDeviceTypes)
          gangDeviceTypes.push_back(crtDeviceTypeAttr);
      }
    } else if (std::get_if<Fortran::parser::AccClause::Vector>(&clause.u)) {
      for (auto crtDeviceTypeAttr : crtDeviceTypes)
        vectorDeviceTypes.push_back(crtDeviceTypeAttr);
    } else if (std::get_if<Fortran::parser::AccClause::Worker>(&clause.u)) {
      for (auto crtDeviceTypeAttr : crtDeviceTypes)
        workerDeviceTypes.push_back(crtDeviceTypeAttr);
    } else if (std::get_if<Fortran::parser::AccClause::Nohost>(&clause.u)) {
      hasNohost = true;
    } else if (const auto *bindClause =
                   std::get_if<Fortran::parser::AccClause::Bind>(&clause.u)) {
      if (const auto *name =
              std::get_if<Fortran::parser::Name>(&bindClause->v.u)) {
        mlir::Attribute bindNameAttr =
            builder.getStringAttr(converter.mangleName(*name->symbol));
        for (auto crtDeviceTypeAttr : crtDeviceTypes) {
          bindNames.push_back(bindNameAttr);
          bindNameDeviceTypes.push_back(crtDeviceTypeAttr);
        }
      } else if (const auto charExpr =
                     std::get_if<Fortran::parser::ScalarDefaultCharExpr>(
                         &bindClause->v.u)) {
        const std::optional<std::string> name =
            Fortran::semantics::GetConstExpr<std::string>(semanticsContext,
                                                          *charExpr);
        if (!name)
          mlir::emitError(loc, "Could not retrieve the bind name");

        mlir::Attribute bindNameAttr = builder.getStringAttr(*name);
        for (auto crtDeviceTypeAttr : crtDeviceTypes) {
          bindNames.push_back(bindNameAttr);
          bindNameDeviceTypes.push_back(crtDeviceTypeAttr);
        }
      }
    } else if (const auto *deviceTypeClause =
                   std::get_if<Fortran::parser::AccClause::DeviceType>(
                       &clause.u)) {
      crtDeviceTypes.clear();
      gatherDeviceTypeAttrs(builder, deviceTypeClause, crtDeviceTypes);
    }
  }

  mlir::OpBuilder modBuilder(mod.getBodyRegion());
  std::stringstream routineOpName;
  routineOpName << accRoutinePrefix.str() << routineCounter++;

  for (auto routineOp : mod.getOps<mlir::acc::RoutineOp>()) {
    if (routineOp.getFuncName().str().compare(funcName) == 0) {
      // If the routine is already specified with the same clauses, just skip
      // the operation creation.
      if (compareDeviceTypeInfo(routineOp, bindNames, bindNameDeviceTypes,
                                gangDeviceTypes, gangDimValues,
                                gangDimDeviceTypes, seqDeviceTypes,
                                workerDeviceTypes, vectorDeviceTypes) &&
          routineOp.getNohost() == hasNohost)
        return;
      mlir::emitError(loc, "Routine already specified with different clauses");
    }
  }

  modBuilder.create<mlir::acc::RoutineOp>(
      loc, routineOpName.str(), funcName,
      bindNames.empty() ? nullptr : builder.getArrayAttr(bindNames),
      bindNameDeviceTypes.empty() ? nullptr
                                  : builder.getArrayAttr(bindNameDeviceTypes),
      workerDeviceTypes.empty() ? nullptr
                                : builder.getArrayAttr(workerDeviceTypes),
      vectorDeviceTypes.empty() ? nullptr
                                : builder.getArrayAttr(vectorDeviceTypes),
      seqDeviceTypes.empty() ? nullptr : builder.getArrayAttr(seqDeviceTypes),
      hasNohost, /*implicit=*/false,
      gangDeviceTypes.empty() ? nullptr : builder.getArrayAttr(gangDeviceTypes),
      gangDimValues.empty() ? nullptr : builder.getArrayAttr(gangDimValues),
      gangDimDeviceTypes.empty() ? nullptr
                                 : builder.getArrayAttr(gangDimDeviceTypes));

  if (funcOp)
    attachRoutineInfo(funcOp, builder.getSymbolRefAttr(routineOpName.str()));
  else
    // FuncOp is not lowered yet. Keep the information so the routine info
    // can be attached later to the funcOp.
    accRoutineInfos.push_back(std::make_pair(
        funcName, builder.getSymbolRefAttr(routineOpName.str())));
}

void Fortran::lower::finalizeOpenACCRoutineAttachment(
    mlir::ModuleOp &mod,
    Fortran::lower::AccRoutineInfoMappingList &accRoutineInfos) {
  for (auto &mapping : accRoutineInfos) {
    mlir::func::FuncOp funcOp =
        mod.lookupSymbol<mlir::func::FuncOp>(mapping.first);
    if (!funcOp)
      mlir::emitWarning(mod.getLoc(),
                        llvm::Twine("function '") + llvm::Twine(mapping.first) +
                            llvm::Twine("' in acc routine directive is not "
                                        "found in this translation unit."));
    else
      attachRoutineInfo(funcOp, mapping.second);
  }
  accRoutineInfos.clear();
}

static void
genACC(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenACCAtomicConstruct &atomicConstruct) {

  mlir::Location loc = converter.genLocation(atomicConstruct.source);
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::AccAtomicRead &atomicRead) {
            Fortran::lower::genOmpAccAtomicRead<Fortran::parser::AccAtomicRead,
                                                void>(converter, atomicRead,
                                                      loc);
          },
          [&](const Fortran::parser::AccAtomicWrite &atomicWrite) {
            Fortran::lower::genOmpAccAtomicWrite<
                Fortran::parser::AccAtomicWrite, void>(converter, atomicWrite,
                                                       loc);
          },
          [&](const Fortran::parser::AccAtomicUpdate &atomicUpdate) {
            Fortran::lower::genOmpAccAtomicUpdate<
                Fortran::parser::AccAtomicUpdate, void>(converter, atomicUpdate,
                                                        loc);
          },
          [&](const Fortran::parser::AccAtomicCapture &atomicCapture) {
            Fortran::lower::genOmpAccAtomicCapture<
                Fortran::parser::AccAtomicCapture, void>(converter,
                                                         atomicCapture, loc);
          },
      },
      atomicConstruct.u);
}

static void
genACC(Fortran::lower::AbstractConverter &converter,
       Fortran::semantics::SemanticsContext &semanticsContext,
       const Fortran::parser::OpenACCCacheConstruct &cacheConstruct) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  auto loopOp = builder.getRegion().getParentOfType<mlir::acc::LoopOp>();
  auto crtPos = builder.saveInsertionPoint();
  if (loopOp) {
    builder.setInsertionPoint(loopOp);
    Fortran::lower::StatementContext stmtCtx;
    llvm::SmallVector<mlir::Value> cacheOperands;
    const Fortran::parser::AccObjectListWithModifier &listWithModifier =
        std::get<Fortran::parser::AccObjectListWithModifier>(cacheConstruct.t);
    const auto &accObjectList =
        std::get<Fortran::parser::AccObjectList>(listWithModifier.t);
    const auto &modifier =
        std::get<std::optional<Fortran::parser::AccDataModifier>>(
            listWithModifier.t);

    mlir::acc::DataClause dataClause = mlir::acc::DataClause::acc_cache;
    if (modifier &&
        (*modifier).v == Fortran::parser::AccDataModifier::Modifier::ReadOnly)
      dataClause = mlir::acc::DataClause::acc_cache_readonly;
    genDataOperandOperations<mlir::acc::CacheOp>(
        accObjectList, converter, semanticsContext, stmtCtx, cacheOperands,
        dataClause,
        /*structured=*/true, /*implicit=*/false, /*setDeclareAttr*/ false);
    loopOp.getCacheOperandsMutable().append(cacheOperands);
  } else {
    llvm::report_fatal_error(
        "could not find loop to attach OpenACC cache information.");
  }
  builder.restoreInsertionPoint(crtPos);
}

mlir::Value Fortran::lower::genOpenACCConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semanticsContext,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenACCConstruct &accConstruct) {

  mlir::Value exitCond;
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
            exitCond = genACC(converter, semanticsContext, eval, loopConstruct);
          },
          [&](const Fortran::parser::OpenACCStandaloneConstruct
                  &standaloneConstruct) {
            genACC(converter, semanticsContext, standaloneConstruct);
          },
          [&](const Fortran::parser::OpenACCCacheConstruct &cacheConstruct) {
            genACC(converter, semanticsContext, cacheConstruct);
          },
          [&](const Fortran::parser::OpenACCWaitConstruct &waitConstruct) {
            genACC(converter, waitConstruct);
          },
          [&](const Fortran::parser::OpenACCAtomicConstruct &atomicConstruct) {
            genACC(converter, eval, atomicConstruct);
          },
          [&](const Fortran::parser::OpenACCEndConstruct &) {
            // No op
          },
      },
      accConstruct.u);
  return exitCond;
}

void Fortran::lower::genOpenACCDeclarativeConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semanticsContext,
    Fortran::lower::StatementContext &openAccCtx,
    const Fortran::parser::OpenACCDeclarativeConstruct &accDeclConstruct,
    Fortran::lower::AccRoutineInfoMappingList &accRoutineInfos) {

  std::visit(
      common::visitors{
          [&](const Fortran::parser::OpenACCStandaloneDeclarativeConstruct
                  &standaloneDeclarativeConstruct) {
            genACC(converter, semanticsContext, openAccCtx,
                   standaloneDeclarativeConstruct);
          },
          [&](const Fortran::parser::OpenACCRoutineConstruct
                  &routineConstruct) {
            fir::FirOpBuilder &builder = converter.getFirOpBuilder();
            mlir::ModuleOp mod = builder.getModule();
            Fortran::lower::genOpenACCRoutineConstruct(
                converter, semanticsContext, mod, routineConstruct,
                accRoutineInfos);
          },
      },
      accDeclConstruct.u);
}

void Fortran::lower::attachDeclarePostAllocAction(
    AbstractConverter &converter, fir::FirOpBuilder &builder,
    const Fortran::semantics::Symbol &sym) {
  std::stringstream fctName;
  fctName << converter.mangleName(sym) << declarePostAllocSuffix.str();
  mlir::Operation &op = builder.getInsertionBlock()->back();
  op.setAttr(mlir::acc::getDeclareActionAttrName(),
             mlir::acc::DeclareActionAttr::get(
                 builder.getContext(),
                 /*preAlloc=*/{},
                 /*postAlloc=*/builder.getSymbolRefAttr(fctName.str()),
                 /*preDealloc=*/{}, /*postDealloc=*/{}));
}

void Fortran::lower::attachDeclarePreDeallocAction(
    AbstractConverter &converter, fir::FirOpBuilder &builder,
    mlir::Value beginOpValue, const Fortran::semantics::Symbol &sym) {
  if (!sym.test(Fortran::semantics::Symbol::Flag::AccCreate) &&
      !sym.test(Fortran::semantics::Symbol::Flag::AccCopyIn) &&
      !sym.test(Fortran::semantics::Symbol::Flag::AccCopyInReadOnly) &&
      !sym.test(Fortran::semantics::Symbol::Flag::AccCopy) &&
      !sym.test(Fortran::semantics::Symbol::Flag::AccCopyOut) &&
      !sym.test(Fortran::semantics::Symbol::Flag::AccDeviceResident))
    return;

  std::stringstream fctName;
  fctName << converter.mangleName(sym) << declarePreDeallocSuffix.str();
  beginOpValue.getDefiningOp()->setAttr(
      mlir::acc::getDeclareActionAttrName(),
      mlir::acc::DeclareActionAttr::get(
          builder.getContext(),
          /*preAlloc=*/{}, /*postAlloc=*/{},
          /*preDealloc=*/builder.getSymbolRefAttr(fctName.str()),
          /*postDealloc=*/{}));
}

void Fortran::lower::attachDeclarePostDeallocAction(
    AbstractConverter &converter, fir::FirOpBuilder &builder,
    const Fortran::semantics::Symbol &sym) {
  if (!sym.test(Fortran::semantics::Symbol::Flag::AccCreate) &&
      !sym.test(Fortran::semantics::Symbol::Flag::AccCopyIn) &&
      !sym.test(Fortran::semantics::Symbol::Flag::AccCopyInReadOnly) &&
      !sym.test(Fortran::semantics::Symbol::Flag::AccCopy) &&
      !sym.test(Fortran::semantics::Symbol::Flag::AccCopyOut) &&
      !sym.test(Fortran::semantics::Symbol::Flag::AccDeviceResident))
    return;

  std::stringstream fctName;
  fctName << converter.mangleName(sym) << declarePostDeallocSuffix.str();
  mlir::Operation &op = builder.getInsertionBlock()->back();
  op.setAttr(mlir::acc::getDeclareActionAttrName(),
             mlir::acc::DeclareActionAttr::get(
                 builder.getContext(),
                 /*preAlloc=*/{}, /*postAlloc=*/{}, /*preDealloc=*/{},
                 /*postDealloc=*/builder.getSymbolRefAttr(fctName.str())));
}

void Fortran::lower::genOpenACCTerminator(fir::FirOpBuilder &builder,
                                          mlir::Operation *op,
                                          mlir::Location loc) {
  if (mlir::isa<mlir::acc::ParallelOp, mlir::acc::LoopOp>(op))
    builder.create<mlir::acc::YieldOp>(loc);
  else
    builder.create<mlir::acc::TerminatorOp>(loc);
}

bool Fortran::lower::isInOpenACCLoop(fir::FirOpBuilder &builder) {
  if (builder.getBlock()->getParent()->getParentOfType<mlir::acc::LoopOp>())
    return true;
  return false;
}

void Fortran::lower::setInsertionPointAfterOpenACCLoopIfInside(
    fir::FirOpBuilder &builder) {
  if (auto loopOp =
          builder.getBlock()->getParent()->getParentOfType<mlir::acc::LoopOp>())
    builder.setInsertionPointAfter(loopOp);
}

void Fortran::lower::genEarlyReturnInOpenACCLoop(fir::FirOpBuilder &builder,
                                                 mlir::Location loc) {
  mlir::Value yieldValue =
      builder.createIntegerConstant(loc, builder.getI1Type(), 1);
  builder.create<mlir::acc::YieldOp>(loc, yieldValue);
}

int64_t Fortran::lower::getCollapseValue(
    const Fortran::parser::AccClauseList &clauseList) {
  for (const Fortran::parser::AccClause &clause : clauseList.v) {
    if (const auto *collapseClause =
            std::get_if<Fortran::parser::AccClause::Collapse>(&clause.u)) {
      const parser::AccCollapseArg &arg = collapseClause->v;
      const auto &collapseValue{std::get<parser::ScalarIntConstantExpr>(arg.t)};
      return *Fortran::semantics::GetIntValue(collapseValue);
    }
  }
  return 1;
}
