//===-- ReductionProcessor.cpp ----------------------------------*- C++ -*-===//
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

#include "ReductionProcessor.h"

#include "flang/Lower/AbstractConverter.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Parser/tools.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "llvm/Support/CommandLine.h"

static llvm::cl::opt<bool> forceByrefReduction(
    "force-byref-reduction",
    llvm::cl::desc("Pass all reduction arguments by reference"),
    llvm::cl::Hidden);

namespace Fortran {
namespace lower {
namespace omp {

ReductionProcessor::ReductionIdentifier ReductionProcessor::getReductionType(
    const omp::clause::ProcedureDesignator &pd) {
  auto redType = llvm::StringSwitch<std::optional<ReductionIdentifier>>(
                     getRealName(pd.v.id()).ToString())
                     .Case("max", ReductionIdentifier::MAX)
                     .Case("min", ReductionIdentifier::MIN)
                     .Case("iand", ReductionIdentifier::IAND)
                     .Case("ior", ReductionIdentifier::IOR)
                     .Case("ieor", ReductionIdentifier::IEOR)
                     .Default(std::nullopt);
  assert(redType && "Invalid Reduction");
  return *redType;
}

ReductionProcessor::ReductionIdentifier ReductionProcessor::getReductionType(
    omp::clause::DefinedOperator::IntrinsicOperator intrinsicOp) {
  switch (intrinsicOp) {
  case omp::clause::DefinedOperator::IntrinsicOperator::Add:
    return ReductionIdentifier::ADD;
  case omp::clause::DefinedOperator::IntrinsicOperator::Subtract:
    return ReductionIdentifier::SUBTRACT;
  case omp::clause::DefinedOperator::IntrinsicOperator::Multiply:
    return ReductionIdentifier::MULTIPLY;
  case omp::clause::DefinedOperator::IntrinsicOperator::AND:
    return ReductionIdentifier::AND;
  case omp::clause::DefinedOperator::IntrinsicOperator::EQV:
    return ReductionIdentifier::EQV;
  case omp::clause::DefinedOperator::IntrinsicOperator::OR:
    return ReductionIdentifier::OR;
  case omp::clause::DefinedOperator::IntrinsicOperator::NEQV:
    return ReductionIdentifier::NEQV;
  default:
    llvm_unreachable("unexpected intrinsic operator in reduction");
  }
}

bool ReductionProcessor::supportedIntrinsicProcReduction(
    const omp::clause::ProcedureDesignator &pd) {
  Fortran::semantics::Symbol *sym = pd.v.id();
  if (!sym->GetUltimate().attrs().test(Fortran::semantics::Attr::INTRINSIC))
    return false;
  auto redType = llvm::StringSwitch<bool>(getRealName(sym).ToString())
                     .Case("max", true)
                     .Case("min", true)
                     .Case("iand", true)
                     .Case("ior", true)
                     .Case("ieor", true)
                     .Default(false);
  return redType;
}

std::string
ReductionProcessor::getReductionName(llvm::StringRef name,
                                     const fir::KindMapping &kindMap,
                                     mlir::Type ty, bool isByRef) {
  ty = fir::unwrapRefType(ty);

  // extra string to distinguish reduction functions for variables passed by
  // reference
  llvm::StringRef byrefAddition{""};
  if (isByRef)
    byrefAddition = "_byref";

  return fir::getTypeAsString(ty, kindMap, (name + byrefAddition).str());
}

std::string ReductionProcessor::getReductionName(
    omp::clause::DefinedOperator::IntrinsicOperator intrinsicOp,
    const fir::KindMapping &kindMap, mlir::Type ty, bool isByRef) {
  std::string reductionName;

  switch (intrinsicOp) {
  case omp::clause::DefinedOperator::IntrinsicOperator::Add:
    reductionName = "add_reduction";
    break;
  case omp::clause::DefinedOperator::IntrinsicOperator::Multiply:
    reductionName = "multiply_reduction";
    break;
  case omp::clause::DefinedOperator::IntrinsicOperator::AND:
    return "and_reduction";
  case omp::clause::DefinedOperator::IntrinsicOperator::EQV:
    return "eqv_reduction";
  case omp::clause::DefinedOperator::IntrinsicOperator::OR:
    return "or_reduction";
  case omp::clause::DefinedOperator::IntrinsicOperator::NEQV:
    return "neqv_reduction";
  default:
    reductionName = "other_reduction";
    break;
  }

  return getReductionName(reductionName, kindMap, ty, isByRef);
}

mlir::Value
ReductionProcessor::getReductionInitValue(mlir::Location loc, mlir::Type type,
                                          ReductionIdentifier redId,
                                          fir::FirOpBuilder &builder) {
  type = fir::unwrapRefType(type);
  if (!fir::isa_integer(type) && !fir::isa_real(type) &&
      !mlir::isa<fir::LogicalType>(type))
    TODO(loc, "Reduction of some types is not supported");
  switch (redId) {
  case ReductionIdentifier::MAX: {
    if (auto ty = type.dyn_cast<mlir::FloatType>()) {
      const llvm::fltSemantics &sem = ty.getFloatSemantics();
      return builder.createRealConstant(
          loc, type, llvm::APFloat::getLargest(sem, /*Negative=*/true));
    }
    unsigned bits = type.getIntOrFloatBitWidth();
    int64_t minInt = llvm::APInt::getSignedMinValue(bits).getSExtValue();
    return builder.createIntegerConstant(loc, type, minInt);
  }
  case ReductionIdentifier::MIN: {
    if (auto ty = type.dyn_cast<mlir::FloatType>()) {
      const llvm::fltSemantics &sem = ty.getFloatSemantics();
      return builder.createRealConstant(
          loc, type, llvm::APFloat::getLargest(sem, /*Negative=*/false));
    }
    unsigned bits = type.getIntOrFloatBitWidth();
    int64_t maxInt = llvm::APInt::getSignedMaxValue(bits).getSExtValue();
    return builder.createIntegerConstant(loc, type, maxInt);
  }
  case ReductionIdentifier::IOR: {
    unsigned bits = type.getIntOrFloatBitWidth();
    int64_t zeroInt = llvm::APInt::getZero(bits).getSExtValue();
    return builder.createIntegerConstant(loc, type, zeroInt);
  }
  case ReductionIdentifier::IEOR: {
    unsigned bits = type.getIntOrFloatBitWidth();
    int64_t zeroInt = llvm::APInt::getZero(bits).getSExtValue();
    return builder.createIntegerConstant(loc, type, zeroInt);
  }
  case ReductionIdentifier::IAND: {
    unsigned bits = type.getIntOrFloatBitWidth();
    int64_t allOnInt = llvm::APInt::getAllOnes(bits).getSExtValue();
    return builder.createIntegerConstant(loc, type, allOnInt);
  }
  case ReductionIdentifier::ADD:
  case ReductionIdentifier::MULTIPLY:
  case ReductionIdentifier::AND:
  case ReductionIdentifier::OR:
  case ReductionIdentifier::EQV:
  case ReductionIdentifier::NEQV:
    if (type.isa<mlir::FloatType>())
      return builder.create<mlir::arith::ConstantOp>(
          loc, type,
          builder.getFloatAttr(type, (double)getOperationIdentity(redId, loc)));

    if (type.isa<fir::LogicalType>()) {
      mlir::Value intConst = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getI1Type(),
          builder.getIntegerAttr(builder.getI1Type(),
                                 getOperationIdentity(redId, loc)));
      return builder.createConvert(loc, type, intConst);
    }

    return builder.create<mlir::arith::ConstantOp>(
        loc, type,
        builder.getIntegerAttr(type, getOperationIdentity(redId, loc)));
  case ReductionIdentifier::ID:
  case ReductionIdentifier::USER_DEF_OP:
  case ReductionIdentifier::SUBTRACT:
    TODO(loc, "Reduction of some identifier types is not supported");
  }
  llvm_unreachable("Unhandled Reduction identifier : getReductionInitValue");
}

mlir::Value ReductionProcessor::createScalarCombiner(
    fir::FirOpBuilder &builder, mlir::Location loc, ReductionIdentifier redId,
    mlir::Type type, mlir::Value op1, mlir::Value op2) {
  mlir::Value reductionOp;
  type = fir::unwrapRefType(type);
  switch (redId) {
  case ReductionIdentifier::MAX:
    reductionOp =
        getReductionOperation<mlir::arith::MaximumFOp, mlir::arith::MaxSIOp>(
            builder, type, loc, op1, op2);
    break;
  case ReductionIdentifier::MIN:
    reductionOp =
        getReductionOperation<mlir::arith::MinimumFOp, mlir::arith::MinSIOp>(
            builder, type, loc, op1, op2);
    break;
  case ReductionIdentifier::IOR:
    assert((type.isIntOrIndex()) && "only integer is expected");
    reductionOp = builder.create<mlir::arith::OrIOp>(loc, op1, op2);
    break;
  case ReductionIdentifier::IEOR:
    assert((type.isIntOrIndex()) && "only integer is expected");
    reductionOp = builder.create<mlir::arith::XOrIOp>(loc, op1, op2);
    break;
  case ReductionIdentifier::IAND:
    assert((type.isIntOrIndex()) && "only integer is expected");
    reductionOp = builder.create<mlir::arith::AndIOp>(loc, op1, op2);
    break;
  case ReductionIdentifier::ADD:
    reductionOp =
        getReductionOperation<mlir::arith::AddFOp, mlir::arith::AddIOp>(
            builder, type, loc, op1, op2);
    break;
  case ReductionIdentifier::MULTIPLY:
    reductionOp =
        getReductionOperation<mlir::arith::MulFOp, mlir::arith::MulIOp>(
            builder, type, loc, op1, op2);
    break;
  case ReductionIdentifier::AND: {
    mlir::Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    mlir::Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    mlir::Value andiOp = builder.create<mlir::arith::AndIOp>(loc, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, andiOp);
    break;
  }
  case ReductionIdentifier::OR: {
    mlir::Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    mlir::Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    mlir::Value oriOp = builder.create<mlir::arith::OrIOp>(loc, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, oriOp);
    break;
  }
  case ReductionIdentifier::EQV: {
    mlir::Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    mlir::Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    mlir::Value cmpiOp = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, cmpiOp);
    break;
  }
  case ReductionIdentifier::NEQV: {
    mlir::Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    mlir::Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    mlir::Value cmpiOp = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, cmpiOp);
    break;
  }
  default:
    TODO(loc, "Reduction of some intrinsic operators is not supported");
  }

  return reductionOp;
}

/// Create reduction combiner region for reduction variables which are boxed
/// arrays
static void genBoxCombiner(fir::FirOpBuilder &builder, mlir::Location loc,
                           ReductionProcessor::ReductionIdentifier redId,
                           fir::BaseBoxType boxTy, mlir::Value lhs,
                           mlir::Value rhs) {
  fir::SequenceType seqTy =
      mlir::dyn_cast_or_null<fir::SequenceType>(boxTy.getEleTy());
  // TODO: support allocatable arrays: !fir.box<!fir.heap<!fir.array<...>>>
  if (!seqTy || seqTy.hasUnknownShape())
    TODO(loc, "Unsupported boxed type in OpenMP reduction");

  // load fir.ref<fir.box<...>>
  mlir::Value lhsAddr = lhs;
  lhs = builder.create<fir::LoadOp>(loc, lhs);
  rhs = builder.create<fir::LoadOp>(loc, rhs);

  const unsigned rank = seqTy.getDimension();
  llvm::SmallVector<mlir::Value> extents;
  extents.reserve(rank);
  llvm::SmallVector<mlir::Value> lbAndExtents;
  lbAndExtents.reserve(rank * 2);

  // Get box lowerbounds and extents:
  mlir::Type idxTy = builder.getIndexType();
  for (unsigned i = 0; i < rank; ++i) {
    // TODO: ideally we want to hoist box reads out of the critical section.
    // We could do this by having box dimensions in block arguments like
    // OpenACC does
    mlir::Value dim = builder.createIntegerConstant(loc, idxTy, i);
    auto dimInfo =
        builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, lhs, dim);
    extents.push_back(dimInfo.getExtent());
    lbAndExtents.push_back(dimInfo.getLowerBound());
    lbAndExtents.push_back(dimInfo.getExtent());
  }

  auto shapeShiftTy = fir::ShapeShiftType::get(builder.getContext(), rank);
  auto shapeShift =
      builder.create<fir::ShapeShiftOp>(loc, shapeShiftTy, lbAndExtents);

  // Iterate over array elements, applying the equivalent scalar reduction:

  // A hlfir::elemental here gets inlined with a temporary so create the
  // loop nest directly.
  // This function already controls all of the code in this region so we
  // know this won't miss any opportuinties for clever elemental inlining
  hlfir::LoopNest nest =
      hlfir::genLoopNest(loc, builder, extents, /*isUnordered=*/true);
  builder.setInsertionPointToStart(nest.innerLoop.getBody());
  mlir::Type refTy = fir::ReferenceType::get(seqTy.getEleTy());
  auto lhsEleAddr = builder.create<fir::ArrayCoorOp>(
      loc, refTy, lhs, shapeShift, /*slice=*/mlir::Value{},
      nest.oneBasedIndices, /*typeparms=*/mlir::ValueRange{});
  auto rhsEleAddr = builder.create<fir::ArrayCoorOp>(
      loc, refTy, rhs, shapeShift, /*slice=*/mlir::Value{},
      nest.oneBasedIndices, /*typeparms=*/mlir::ValueRange{});
  auto lhsEle = builder.create<fir::LoadOp>(loc, lhsEleAddr);
  auto rhsEle = builder.create<fir::LoadOp>(loc, rhsEleAddr);
  mlir::Value scalarReduction = ReductionProcessor::createScalarCombiner(
      builder, loc, redId, refTy, lhsEle, rhsEle);
  builder.create<fir::StoreOp>(loc, scalarReduction, lhsEleAddr);

  builder.setInsertionPointAfter(nest.outerLoop);
  builder.create<mlir::omp::YieldOp>(loc, lhsAddr);
}

// generate combiner region for reduction operations
static void genCombiner(fir::FirOpBuilder &builder, mlir::Location loc,
                        ReductionProcessor::ReductionIdentifier redId,
                        mlir::Type ty, mlir::Value lhs, mlir::Value rhs,
                        bool isByRef) {
  ty = fir::unwrapRefType(ty);

  if (fir::isa_trivial(ty)) {
    mlir::Value lhsLoaded = builder.loadIfRef(loc, lhs);
    mlir::Value rhsLoaded = builder.loadIfRef(loc, rhs);

    mlir::Value result = ReductionProcessor::createScalarCombiner(
        builder, loc, redId, ty, lhsLoaded, rhsLoaded);
    if (isByRef) {
      builder.create<fir::StoreOp>(loc, result, lhs);
      builder.create<mlir::omp::YieldOp>(loc, lhs);
    } else {
      builder.create<mlir::omp::YieldOp>(loc, result);
    }
    return;
  }
  // all arrays should have been boxed
  if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(ty)) {
    genBoxCombiner(builder, loc, redId, boxTy, lhs, rhs);
    return;
  }

  TODO(loc, "OpenMP genCombiner for unsupported reduction variable type");
}

static mlir::Value
createReductionInitRegion(fir::FirOpBuilder &builder, mlir::Location loc,
                          const ReductionProcessor::ReductionIdentifier redId,
                          mlir::Type type, bool isByRef) {
  mlir::Type ty = fir::unwrapRefType(type);
  mlir::Value initValue = ReductionProcessor::getReductionInitValue(
      loc, fir::unwrapSeqOrBoxedSeqType(ty), redId, builder);

  if (fir::isa_trivial(ty)) {
    if (isByRef) {
      mlir::Value alloca = builder.create<fir::AllocaOp>(loc, ty);
      builder.createStoreWithConvert(loc, initValue, alloca);
      return alloca;
    }
    // by val
    return initValue;
  }

  // all arrays are boxed
  if (auto boxTy = mlir::dyn_cast_or_null<fir::BaseBoxType>(ty)) {
    assert(isByRef && "passing arrays by value is unsupported");
    // TODO: support allocatable arrays: !fir.box<!fir.heap<!fir.array<...>>>
    mlir::Type innerTy = fir::extractSequenceType(boxTy);
    if (!mlir::isa<fir::SequenceType>(innerTy))
      TODO(loc, "Unsupported boxed type for reduction");
    // Create the private copy from the initial fir.box:
    hlfir::Entity source = hlfir::Entity{builder.getBlock()->getArgument(0)};

    // TODO: if the whole reduction is nested inside of a loop, this alloca
    // could lead to a stack overflow (the memory is only freed at the end of
    // the stack frame). The reduction declare operation needs a deallocation
    // region to undo the init region.
    hlfir::Entity temp = createStackTempFromMold(loc, builder, source);

    // Put the temporary inside of a box:
    hlfir::Entity box = hlfir::genVariableBox(loc, builder, temp);
    builder.create<hlfir::AssignOp>(loc, initValue, box);
    mlir::Value boxAlloca = builder.create<fir::AllocaOp>(loc, ty);
    builder.create<fir::StoreOp>(loc, box, boxAlloca);
    return boxAlloca;
  }

  TODO(loc, "createReductionInitRegion for unsupported type");
}

mlir::omp::DeclareReductionOp ReductionProcessor::createDeclareReduction(
    fir::FirOpBuilder &builder, llvm::StringRef reductionOpName,
    const ReductionIdentifier redId, mlir::Type type, mlir::Location loc,
    bool isByRef) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::ModuleOp module = builder.getModule();

  assert(!reductionOpName.empty());

  auto decl =
      module.lookupSymbol<mlir::omp::DeclareReductionOp>(reductionOpName);
  if (decl)
    return decl;

  mlir::OpBuilder modBuilder(module.getBodyRegion());
  mlir::Type valTy = fir::unwrapRefType(type);
  if (!isByRef)
    type = valTy;

  decl = modBuilder.create<mlir::omp::DeclareReductionOp>(loc, reductionOpName,
                                                          type);
  builder.createBlock(&decl.getInitializerRegion(),
                      decl.getInitializerRegion().end(), {type}, {loc});
  builder.setInsertionPointToEnd(&decl.getInitializerRegion().back());

  mlir::Value init =
      createReductionInitRegion(builder, loc, redId, type, isByRef);
  builder.create<mlir::omp::YieldOp>(loc, init);

  builder.createBlock(&decl.getReductionRegion(),
                      decl.getReductionRegion().end(), {type, type},
                      {loc, loc});

  builder.setInsertionPointToEnd(&decl.getReductionRegion().back());
  mlir::Value op1 = decl.getReductionRegion().front().getArgument(0);
  mlir::Value op2 = decl.getReductionRegion().front().getArgument(1);
  genCombiner(builder, loc, redId, type, op1, op2, isByRef);

  return decl;
}

// TODO: By-ref vs by-val reductions are currently toggled for the whole
//       operation (possibly effecting multiple reduction variables).
//       This could cause a problem with openmp target reductions because
//       by-ref trivial types may not be supported.
bool ReductionProcessor::doReductionByRef(
    const llvm::SmallVectorImpl<mlir::Value> &reductionVars) {
  if (reductionVars.empty())
    return false;
  if (forceByrefReduction)
    return true;

  for (mlir::Value reductionVar : reductionVars) {
    if (auto declare =
            mlir::dyn_cast<hlfir::DeclareOp>(reductionVar.getDefiningOp()))
      reductionVar = declare.getMemref();

    if (!fir::isa_trivial(fir::unwrapRefType(reductionVar.getType())))
      return true;
  }
  return false;
}

void ReductionProcessor::addDeclareReduction(
    mlir::Location currentLocation,
    Fortran::lower::AbstractConverter &converter,
    const omp::clause::Reduction &reduction,
    llvm::SmallVectorImpl<mlir::Value> &reductionVars,
    llvm::SmallVectorImpl<mlir::Attribute> &reductionDeclSymbols,
    llvm::SmallVectorImpl<const Fortran::semantics::Symbol *>
        *reductionSymbols) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::omp::DeclareReductionOp decl;
  const auto &redOperator{
      std::get<omp::clause::ReductionOperator>(reduction.t)};
  const auto &objectList{std::get<omp::ObjectList>(reduction.t)};

  if (!std::holds_alternative<omp::clause::DefinedOperator>(redOperator.u)) {
    if (const auto *reductionIntrinsic =
            std::get_if<omp::clause::ProcedureDesignator>(&redOperator.u)) {
      if (!ReductionProcessor::supportedIntrinsicProcReduction(
              *reductionIntrinsic)) {
        return;
      }
    } else {
      return;
    }
  }

  // initial pass to collect all reduction vars so we can figure out if this
  // should happen byref
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  for (const Object &object : objectList) {
    const Fortran::semantics::Symbol *symbol = object.id();
    if (reductionSymbols)
      reductionSymbols->push_back(symbol);
    mlir::Value symVal = converter.getSymbolAddress(*symbol);
    auto redType = mlir::cast<fir::ReferenceType>(symVal.getType());

    // all arrays must be boxed so that we have convenient access to all the
    // information needed to iterate over the array
    if (mlir::isa<fir::SequenceType>(redType.getEleTy())) {
      hlfir::Entity entity{symVal};
      entity = genVariableBox(currentLocation, builder, entity);
      mlir::Value box = entity.getBase();

      // Always pass the box by reference so that the OpenMP dialect
      // verifiers don't need to know anything about fir.box
      auto alloca =
          builder.create<fir::AllocaOp>(currentLocation, box.getType());
      builder.create<fir::StoreOp>(currentLocation, box, alloca);

      symVal = alloca;
      redType = mlir::cast<fir::ReferenceType>(symVal.getType());
    } else if (auto declOp = symVal.getDefiningOp<hlfir::DeclareOp>()) {
      symVal = declOp.getBase();
    }

    reductionVars.push_back(symVal);
  }
  const bool isByRef = doReductionByRef(reductionVars);

  if (const auto &redDefinedOp =
          std::get_if<omp::clause::DefinedOperator>(&redOperator.u)) {
    const auto &intrinsicOp{
        std::get<omp::clause::DefinedOperator::IntrinsicOperator>(
            redDefinedOp->u)};
    ReductionIdentifier redId = getReductionType(intrinsicOp);
    switch (redId) {
    case ReductionIdentifier::ADD:
    case ReductionIdentifier::MULTIPLY:
    case ReductionIdentifier::AND:
    case ReductionIdentifier::EQV:
    case ReductionIdentifier::OR:
    case ReductionIdentifier::NEQV:
      break;
    default:
      TODO(currentLocation,
           "Reduction of some intrinsic operators is not supported");
      break;
    }

    for (mlir::Value symVal : reductionVars) {
      auto redType = mlir::cast<fir::ReferenceType>(symVal.getType());
      const auto &kindMap = firOpBuilder.getKindMap();
      if (redType.getEleTy().isa<fir::LogicalType>())
        decl = createDeclareReduction(firOpBuilder,
                                      getReductionName(intrinsicOp, kindMap,
                                                       firOpBuilder.getI1Type(),
                                                       isByRef),
                                      redId, redType, currentLocation, isByRef);
      else
        decl = createDeclareReduction(
            firOpBuilder,
            getReductionName(intrinsicOp, kindMap, redType, isByRef), redId,
            redType, currentLocation, isByRef);
      reductionDeclSymbols.push_back(mlir::SymbolRefAttr::get(
          firOpBuilder.getContext(), decl.getSymName()));
    }
  } else if (const auto *reductionIntrinsic =
                 std::get_if<omp::clause::ProcedureDesignator>(
                     &redOperator.u)) {
    if (ReductionProcessor::supportedIntrinsicProcReduction(
            *reductionIntrinsic)) {
      ReductionProcessor::ReductionIdentifier redId =
          ReductionProcessor::getReductionType(*reductionIntrinsic);
      for (const Object &object : objectList) {
        const Fortran::semantics::Symbol *symbol = object.id();
        mlir::Value symVal = converter.getSymbolAddress(*symbol);
        if (auto declOp = symVal.getDefiningOp<hlfir::DeclareOp>())
          symVal = declOp.getBase();
        auto redType = symVal.getType().cast<fir::ReferenceType>();
        if (!redType.getEleTy().isIntOrIndexOrFloat())
          TODO(currentLocation, "User Defined Reduction on non-trivial type");
        decl = createDeclareReduction(
            firOpBuilder,
            getReductionName(getRealName(*reductionIntrinsic).ToString(),
                             firOpBuilder.getKindMap(), redType, isByRef),
            redId, redType, currentLocation, isByRef);
        reductionDeclSymbols.push_back(mlir::SymbolRefAttr::get(
            firOpBuilder.getContext(), decl.getSymName()));
      }
    }
  }
}

const Fortran::semantics::SourceName
ReductionProcessor::getRealName(const Fortran::semantics::Symbol *symbol) {
  return symbol->GetUltimate().name();
}

const Fortran::semantics::SourceName
ReductionProcessor::getRealName(const omp::clause::ProcedureDesignator &pd) {
  return getRealName(pd.v.id());
}

int ReductionProcessor::getOperationIdentity(ReductionIdentifier redId,
                                             mlir::Location loc) {
  switch (redId) {
  case ReductionIdentifier::ADD:
  case ReductionIdentifier::OR:
  case ReductionIdentifier::NEQV:
    return 0;
  case ReductionIdentifier::MULTIPLY:
  case ReductionIdentifier::AND:
  case ReductionIdentifier::EQV:
    return 1;
  default:
    TODO(loc, "Reduction of some intrinsic operators is not supported");
  }
}

} // namespace omp
} // namespace lower
} // namespace Fortran
