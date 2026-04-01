//===-- ReductionProcessor.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://aiir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Support/ReductionProcessor.h"

#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/OpenMP/Clauses.h"
#include "flang/Lower/Support/PrivateReductionUtils.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "aiir/Dialect/OpenMP/OpenMPDialect.h"
#include "llvm/Support/CommandLine.h"
#include <type_traits>

static llvm::cl::opt<bool> forceByrefReduction(
    "force-byref-reduction",
    llvm::cl::desc("Pass all reduction arguments by reference"),
    llvm::cl::Hidden);

using ReductionModifier =
    Fortran::lower::omp::clause::Reduction::ReductionModifier;

namespace Fortran {
namespace lower {
namespace omp {

// explicit template declarations
template bool ReductionProcessor::processReductionArguments<
    aiir::omp::DeclareReductionOp, omp::clause::ReductionOperatorList>(
    aiir::Location currentLocation, lower::AbstractConverter &converter,
    const omp::clause::ReductionOperatorList &redOperatorList,
    llvm::SmallVectorImpl<aiir::Value> &reductionVars,
    llvm::SmallVectorImpl<bool> &reduceVarByRef,
    llvm::SmallVectorImpl<aiir::Attribute> &reductionDeclSymbols,
    const llvm::SmallVectorImpl<const semantics::Symbol *> &reductionSymbols);

template bool ReductionProcessor::processReductionArguments<
    fir::DeclareReductionOp, llvm::SmallVector<fir::ReduceOperationEnum>>(
    aiir::Location currentLocation, lower::AbstractConverter &converter,
    const llvm::SmallVector<fir::ReduceOperationEnum> &redOperatorList,
    llvm::SmallVectorImpl<aiir::Value> &reductionVars,
    llvm::SmallVectorImpl<bool> &reduceVarByRef,
    llvm::SmallVectorImpl<aiir::Attribute> &reductionDeclSymbols,
    const llvm::SmallVectorImpl<const semantics::Symbol *> &reductionSymbols);

template aiir::omp::DeclareReductionOp
ReductionProcessor::createDeclareReduction<aiir::omp::DeclareReductionOp>(
    AbstractConverter &converter, llvm::StringRef reductionOpName,
    const ReductionIdentifier redId, aiir::Type type, aiir::Location loc,
    bool isByRef);

template fir::DeclareReductionOp
ReductionProcessor::createDeclareReduction<fir::DeclareReductionOp>(
    AbstractConverter &converter, llvm::StringRef reductionOpName,
    const ReductionIdentifier redId, aiir::Type type, aiir::Location loc,
    bool isByRef);

ReductionProcessor::ReductionIdentifier ReductionProcessor::getReductionType(
    const omp::clause::ProcedureDesignator &pd) {
  auto redType = llvm::StringSwitch<std::optional<ReductionIdentifier>>(
                     getRealName(pd.v.sym()).ToString())
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

ReductionProcessor::ReductionIdentifier
ReductionProcessor::getReductionType(const fir::ReduceOperationEnum &redOp) {
  switch (redOp) {
  case fir::ReduceOperationEnum::Add:
    return ReductionIdentifier::ADD;
  case fir::ReduceOperationEnum::Multiply:
    return ReductionIdentifier::MULTIPLY;

  case fir::ReduceOperationEnum::AND:
    return ReductionIdentifier::AND;
  case fir::ReduceOperationEnum::OR:
    return ReductionIdentifier::OR;

  case fir::ReduceOperationEnum::EQV:
    return ReductionIdentifier::EQV;
  case fir::ReduceOperationEnum::NEQV:
    return ReductionIdentifier::NEQV;

  case fir::ReduceOperationEnum::IAND:
    return ReductionIdentifier::IAND;
  case fir::ReduceOperationEnum::IEOR:
    return ReductionIdentifier::IEOR;
  case fir::ReduceOperationEnum::IOR:
    return ReductionIdentifier::IOR;
  case fir::ReduceOperationEnum::MAX:
    return ReductionIdentifier::MAX;
  case fir::ReduceOperationEnum::MIN:
    return ReductionIdentifier::MIN;
  }
  llvm_unreachable("Unhandled ReductionIdentifier case");
}

bool ReductionProcessor::supportedIntrinsicProcReduction(
    const omp::clause::ProcedureDesignator &pd) {
  semantics::Symbol *sym = pd.v.sym();
  if (!sym->GetUltimate().attrs().test(semantics::Attr::INTRINSIC))
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
                                     aiir::Type ty, bool isByRef) {
  ty = fir::unwrapRefType(ty);

  // extra string to distinguish reduction functions for variables passed by
  // reference
  llvm::StringRef byrefAddition{""};
  if (isByRef)
    byrefAddition = "_byref";

  return fir::getTypeAsString(ty, kindMap, (name + byrefAddition).str());
}

std::string
ReductionProcessor::getReductionName(ReductionIdentifier redId,
                                     const fir::KindMapping &kindMap,
                                     aiir::Type ty, bool isByRef) {
  std::string reductionName;

  switch (redId) {
  case ReductionIdentifier::ADD:
    reductionName = "add_reduction";
    break;
  case ReductionIdentifier::MULTIPLY:
    reductionName = "multiply_reduction";
    break;
  case ReductionIdentifier::AND:
    reductionName = "and_reduction";
    break;
  case ReductionIdentifier::EQV:
    reductionName = "eqv_reduction";
    break;
  case ReductionIdentifier::OR:
    reductionName = "or_reduction";
    break;
  case ReductionIdentifier::NEQV:
    reductionName = "neqv_reduction";
    break;
  default:
    reductionName = "other_reduction";
    break;
  }

  return getReductionName(reductionName, kindMap, ty, isByRef);
}

aiir::Value
ReductionProcessor::getReductionInitValue(aiir::Location loc, aiir::Type type,
                                          ReductionIdentifier redId,
                                          fir::FirOpBuilder &builder) {
  type = fir::unwrapRefType(type);
  if (!fir::isa_integer(type) && !fir::isa_real(type) &&
      !fir::isa_complex(type) && !aiir::isa<fir::LogicalType>(type))
    TODO(loc, "Reduction of some types is not supported");
  switch (redId) {
  case ReductionIdentifier::MAX: {
    if (auto ty = aiir::dyn_cast<aiir::FloatType>(type)) {
      const llvm::fltSemantics &sem = ty.getFloatSemantics();
      return builder.createRealConstant(
          loc, type, llvm::APFloat::getLargest(sem, /*Negative=*/true));
    }
    unsigned bits = type.getIntOrFloatBitWidth();
    int64_t minInt = llvm::APInt::getSignedMinValue(bits).getSExtValue();
    return builder.createIntegerConstant(loc, type, minInt);
  }
  case ReductionIdentifier::MIN: {
    if (auto ty = aiir::dyn_cast<aiir::FloatType>(type)) {
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
    if (auto cplxTy = aiir::dyn_cast<aiir::ComplexType>(type)) {
      aiir::Type realTy = cplxTy.getElementType();
      aiir::Value initRe = builder.createRealConstant(
          loc, realTy, getOperationIdentity(redId, loc));
      aiir::Value initIm = builder.createRealConstant(loc, realTy, 0);

      return fir::factory::Complex{builder, loc}.createComplex(type, initRe,
                                                               initIm);
    }
    if (aiir::isa<aiir::FloatType>(type))
      return aiir::arith::ConstantOp::create(
          builder, loc, type,
          builder.getFloatAttr(type, (double)getOperationIdentity(redId, loc)));

    if (aiir::isa<fir::LogicalType>(type)) {
      aiir::Value intConst = aiir::arith::ConstantOp::create(
          builder, loc, builder.getI1Type(),
          builder.getIntegerAttr(builder.getI1Type(),
                                 getOperationIdentity(redId, loc)));
      return builder.createConvert(loc, type, intConst);
    }

    return aiir::arith::ConstantOp::create(
        builder, loc, type,
        builder.getIntegerAttr(type, getOperationIdentity(redId, loc)));
  case ReductionIdentifier::ID:
  case ReductionIdentifier::USER_DEF_OP:
  case ReductionIdentifier::SUBTRACT:
    TODO(loc, "Reduction of some identifier types is not supported");
  }
  llvm_unreachable("Unhandled Reduction identifier : getReductionInitValue");
}

aiir::Value ReductionProcessor::createScalarCombiner(
    fir::FirOpBuilder &builder, aiir::Location loc, ReductionIdentifier redId,
    aiir::Type type, aiir::Value op1, aiir::Value op2) {
  aiir::Value reductionOp;
  type = fir::unwrapRefType(type);
  switch (redId) {
  case ReductionIdentifier::MAX:
    reductionOp =
        getReductionOperation<aiir::arith::MaxNumFOp, aiir::arith::MaxSIOp>(
            builder, type, loc, op1, op2);
    break;
  case ReductionIdentifier::MIN:
    reductionOp =
        getReductionOperation<aiir::arith::MinNumFOp, aiir::arith::MinSIOp>(
            builder, type, loc, op1, op2);
    break;
  case ReductionIdentifier::IOR:
    assert((type.isIntOrIndex()) && "only integer is expected");
    reductionOp = aiir::arith::OrIOp::create(builder, loc, op1, op2);
    break;
  case ReductionIdentifier::IEOR:
    assert((type.isIntOrIndex()) && "only integer is expected");
    reductionOp = aiir::arith::XOrIOp::create(builder, loc, op1, op2);
    break;
  case ReductionIdentifier::IAND:
    assert((type.isIntOrIndex()) && "only integer is expected");
    reductionOp = aiir::arith::AndIOp::create(builder, loc, op1, op2);
    break;
  case ReductionIdentifier::ADD:
    reductionOp =
        getReductionOperation<aiir::arith::AddFOp, aiir::arith::AddIOp,
                              fir::AddcOp>(builder, type, loc, op1, op2);
    break;
  case ReductionIdentifier::MULTIPLY:
    reductionOp =
        getReductionOperation<aiir::arith::MulFOp, aiir::arith::MulIOp,
                              fir::MulcOp>(builder, type, loc, op1, op2);
    break;
  case ReductionIdentifier::AND: {
    aiir::Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    aiir::Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    aiir::Value andiOp =
        aiir::arith::AndIOp::create(builder, loc, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, andiOp);
    break;
  }
  case ReductionIdentifier::OR: {
    aiir::Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    aiir::Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    aiir::Value oriOp = aiir::arith::OrIOp::create(builder, loc, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, oriOp);
    break;
  }
  case ReductionIdentifier::EQV: {
    aiir::Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    aiir::Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    aiir::Value cmpiOp = aiir::arith::CmpIOp::create(
        builder, loc, aiir::arith::CmpIPredicate::eq, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, cmpiOp);
    break;
  }
  case ReductionIdentifier::NEQV: {
    aiir::Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    aiir::Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    aiir::Value cmpiOp = aiir::arith::CmpIOp::create(
        builder, loc, aiir::arith::CmpIPredicate::ne, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, cmpiOp);
    break;
  }
  default:
    TODO(loc, "Reduction of some intrinsic operators is not supported");
  }

  return reductionOp;
}

template <typename ParentDeclOpType>
static void genYield(fir::FirOpBuilder &builder, aiir::Location loc,
                     aiir::Value yieldedValue) {
  if constexpr (std::is_same_v<ParentDeclOpType, aiir::omp::DeclareReductionOp>)
    aiir::omp::YieldOp::create(builder, loc, yieldedValue);
  else
    fir::YieldOp::create(builder, loc, yieldedValue);
}

/// Create reduction combiner region for reduction variables which are boxed
/// arrays
template <typename DeclRedOpType>
static void genBoxCombiner(fir::FirOpBuilder &builder, aiir::Location loc,
                           ReductionProcessor::ReductionIdentifier redId,
                           fir::BaseBoxType boxTy, aiir::Value lhs,
                           aiir::Value rhs) {
  fir::SequenceType seqTy = aiir::dyn_cast_or_null<fir::SequenceType>(
      fir::unwrapRefType(boxTy.getEleTy()));
  fir::HeapType heapTy =
      aiir::dyn_cast_or_null<fir::HeapType>(boxTy.getEleTy());
  fir::PointerType ptrTy =
      aiir::dyn_cast_or_null<fir::PointerType>(boxTy.getEleTy());
  if ((!seqTy || seqTy.hasUnknownShape()) && !heapTy && !ptrTy)
    TODO(loc, "Unsupported boxed type in OpenMP reduction");

  // load fir.ref<fir.box<...>>
  aiir::Value lhsAddr = lhs;
  lhs = fir::LoadOp::create(builder, loc, lhs);
  rhs = fir::LoadOp::create(builder, loc, rhs);

  if ((heapTy || ptrTy) && !seqTy) {
    // get box contents (heap pointers)
    lhs = fir::BoxAddrOp::create(builder, loc, lhs);
    rhs = fir::BoxAddrOp::create(builder, loc, rhs);
    aiir::Value lhsValAddr = lhs;

    // load heap pointers
    lhs = fir::LoadOp::create(builder, loc, lhs);
    rhs = fir::LoadOp::create(builder, loc, rhs);

    aiir::Type eleTy = heapTy ? heapTy.getEleTy() : ptrTy.getEleTy();

    aiir::Value result = ReductionProcessor::createScalarCombiner(
        builder, loc, redId, eleTy, lhs, rhs);
    fir::StoreOp::create(builder, loc, result, lhsValAddr);
    genYield<DeclRedOpType>(builder, loc, lhsAddr);
    return;
  }

  // Get ShapeShift with default lower bounds. This makes it possible to use
  // unmodified LoopNest's indices with ArrayCoorOp.
  fir::ShapeShiftOp shapeShift =
      getShapeShift(builder, loc, lhs,
                    /*cannotHaveNonDefaultLowerBounds=*/false,
                    /*useDefaultLowerBounds=*/true);

  // Iterate over array elements, applying the equivalent scalar reduction:

  // F2018 5.4.10.2: Unallocated allocatable variables may not be referenced
  // and so no null check is needed here before indexing into the (possibly
  // allocatable) arrays.

  // A hlfir::elemental here gets inlined with a temporary so create the
  // loop nest directly.
  // This function already controls all of the code in this region so we
  // know this won't miss any opportuinties for clever elemental inlining
  hlfir::LoopNest nest = hlfir::genLoopNest(
      loc, builder, shapeShift.getExtents(), /*isUnordered=*/true);
  builder.setInsertionPointToStart(nest.body);
  const bool seqIsVolatile = fir::isa_volatile_type(seqTy.getEleTy());
  aiir::Type refTy = fir::ReferenceType::get(seqTy.getEleTy(), seqIsVolatile);
  auto lhsEleAddr = fir::ArrayCoorOp::create(
      builder, loc, refTy, lhs, shapeShift, /*slice=*/aiir::Value{},
      nest.oneBasedIndices, /*typeparms=*/aiir::ValueRange{});
  auto rhsEleAddr = fir::ArrayCoorOp::create(
      builder, loc, refTy, rhs, shapeShift, /*slice=*/aiir::Value{},
      nest.oneBasedIndices, /*typeparms=*/aiir::ValueRange{});
  auto lhsEle = fir::LoadOp::create(builder, loc, lhsEleAddr);
  auto rhsEle = fir::LoadOp::create(builder, loc, rhsEleAddr);
  aiir::Value scalarReduction = ReductionProcessor::createScalarCombiner(
      builder, loc, redId, refTy, lhsEle, rhsEle);
  fir::StoreOp::create(builder, loc, scalarReduction, lhsEleAddr);

  builder.setInsertionPointAfter(nest.outerOp);
  genYield<DeclRedOpType>(builder, loc, lhsAddr);
}

// generate combiner region for reduction operations
template <typename DeclRedOpType>
static void genCombiner(fir::FirOpBuilder &builder, aiir::Location loc,
                        ReductionProcessor::ReductionIdentifier redId,
                        aiir::Type ty, aiir::Value lhs, aiir::Value rhs,
                        bool isByRef) {
  ty = fir::unwrapRefType(ty);

  if (fir::isa_trivial(ty)) {
    aiir::Value lhsLoaded = builder.loadIfRef(loc, lhs);
    aiir::Value rhsLoaded = builder.loadIfRef(loc, rhs);

    aiir::Value result = ReductionProcessor::createScalarCombiner(
        builder, loc, redId, ty, lhsLoaded, rhsLoaded);
    if (isByRef) {
      fir::StoreOp::create(builder, loc, result, lhs);
      genYield<DeclRedOpType>(builder, loc, lhs);
    } else {
      genYield<DeclRedOpType>(builder, loc, result);
    }
    return;
  }
  // all arrays should have been boxed
  if (auto boxTy = aiir::dyn_cast<fir::BaseBoxType>(ty)) {
    genBoxCombiner<DeclRedOpType>(builder, loc, redId, boxTy, lhs, rhs);
    return;
  }

  TODO(loc, "OpenMP genCombiner for unsupported reduction variable type");
}

// like fir::unwrapSeqOrBoxedSeqType except it also works for non-sequence boxes
static aiir::Type unwrapSeqOrBoxedType(aiir::Type ty) {
  if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(ty))
    return seqTy.getEleTy();
  if (auto boxTy = aiir::dyn_cast<fir::BaseBoxType>(ty)) {
    auto eleTy = fir::unwrapRefType(boxTy.getEleTy());
    if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(eleTy))
      return seqTy.getEleTy();
    return eleTy;
  }
  return ty;
}

template <typename OpType>
static void createReductionAllocAndInitRegions(
    AbstractConverter &converter, aiir::Location loc, OpType &reductionDecl,
    ReductionProcessor::GenInitValueCBTy genInitValueCB, aiir::Type type,
    bool isByRef) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  auto yield = [&](aiir::Value ret) { genYield<OpType>(builder, loc, ret); };

  aiir::Block *allocBlock = nullptr;
  aiir::Block *initBlock = nullptr;
  if (isByRef) {
    allocBlock =
        builder.createBlock(&reductionDecl.getAllocRegion(),
                            reductionDecl.getAllocRegion().end(), {}, {});
    initBlock = builder.createBlock(&reductionDecl.getInitializerRegion(),
                                    reductionDecl.getInitializerRegion().end(),
                                    {type, type}, {loc, loc});
  } else {
    initBlock = builder.createBlock(&reductionDecl.getInitializerRegion(),
                                    reductionDecl.getInitializerRegion().end(),
                                    {type}, {loc});
  }

  aiir::Type ty = fir::unwrapRefType(type);
  builder.setInsertionPointToEnd(initBlock);
  aiir::Value initValue =
      genInitValueCB(builder, loc, ty, initBlock->getArgument(0));
  if (isByRef) {
    populateByRefInitAndCleanupRegions(
        converter, loc, type, initValue, initBlock,
        reductionDecl.getInitializerAllocArg(),
        reductionDecl.getInitializerMoldArg(), reductionDecl.getCleanupRegion(),
        DeclOperationKind::Reduction, /*sym=*/nullptr,
        /*cannotHaveLowerBounds=*/false,
        /*isDoConcurrent*/ std::is_same_v<OpType, fir::DeclareReductionOp>);
  }

  if (fir::isa_trivial(ty) || fir::isa_derived(ty)) {
    if (isByRef) {
      // alloc region
      builder.setInsertionPointToEnd(allocBlock);
      aiir::Value alloca = fir::AllocaOp::create(builder, loc, ty);
      yield(alloca);
      return;
    }
    // by val
    yield(initValue);
    return;
  }
  assert(isByRef && "passing non-trivial types by val is unsupported");

  // alloc region
  builder.setInsertionPointToEnd(allocBlock);
  aiir::Value boxAlloca = fir::AllocaOp::create(builder, loc, ty);
  yield(boxAlloca);
}

template <typename DeclareRedType>
DeclareRedType ReductionProcessor::createDeclareReductionHelper(
    AbstractConverter &converter, llvm::StringRef reductionOpName,
    aiir::Type type, aiir::Location loc, bool isByRef,
    GenCombinerCBTy genCombinerCB, GenInitValueCBTy genInitValueCB) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  aiir::OpBuilder::InsertionGuard guard(builder);
  aiir::ModuleOp module = builder.getModule();

  assert(!reductionOpName.empty());

  auto decl = module.lookupSymbol<DeclareRedType>(reductionOpName);
  if (decl)
    return decl;

  aiir::OpBuilder modBuilder(module.getBodyRegion());
  aiir::Type valTy = fir::unwrapRefType(type);

  // For by-ref reductions, we want to keep track of the
  // boxed/referenced/allocated type. For example, for a `real, allocatable`
  // variable, `real` should be stored.
  aiir::TypeAttr boxedTyAttr{};
  aiir::Type boxedTy;

  if (isByRef) {
    boxedTy = fir::unwrapPassByRefType(valTy);
    boxedTyAttr = aiir::TypeAttr::get(boxedTy);
    // For character types that are not already references, we need to wrap
    // them in a reference type for by-ref reductions.
    if (fir::isa_char(valTy) && !fir::isa_ref_type(type)) {
      type = fir::ReferenceType::get(valTy);
    }
  } else
    type = valTy;

  decl = DeclareRedType::create(modBuilder, loc, reductionOpName, type,
                                boxedTyAttr);
  createReductionAllocAndInitRegions(converter, loc, decl, genInitValueCB, type,
                                     isByRef);
  builder.createBlock(&decl.getReductionRegion(),
                      decl.getReductionRegion().end(), {type, type},
                      {loc, loc});
  builder.setInsertionPointToEnd(&decl.getReductionRegion().back());
  aiir::Value op1 = decl.getReductionRegion().front().getArgument(0);
  aiir::Value op2 = decl.getReductionRegion().front().getArgument(1);
  genCombinerCB(builder, loc, type, op1, op2, isByRef);

  if (isByRef && fir::isa_box_type(valTy)) {
    aiir::Region &dataPtrPtrRegion = decl.getDataPtrPtrRegion();
    aiir::Block &dataAddrBlock = *builder.createBlock(
        &dataPtrPtrRegion, dataPtrPtrRegion.end(), {type}, {loc});
    builder.setInsertionPointToEnd(&dataAddrBlock);
    aiir::Value boxRefOperand = dataAddrBlock.getArgument(0);
    aiir::Value baseAddrOffset = fir::BoxOffsetOp::create(
        builder, loc, boxRefOperand, fir::BoxFieldAttr::base_addr);
    genYield<DeclareRedType>(builder, loc, baseAddrOffset);
  }

  return decl;
}

template <typename OpType>
OpType ReductionProcessor::createDeclareReduction(
    AbstractConverter &converter, llvm::StringRef reductionOpName,
    const ReductionIdentifier redId, aiir::Type type, aiir::Location loc,
    bool isByRef) {
  auto genInitValueCB = [&](fir::FirOpBuilder &builder, aiir::Location loc,
                            aiir::Type type, aiir::Value val) {
    aiir::Type ty = fir::unwrapRefType(type);
    aiir::Value initValue = ReductionProcessor::getReductionInitValue(
        loc, unwrapSeqOrBoxedType(ty), redId, builder);
    return initValue;
  };
  auto genCombinerCB = [&](fir::FirOpBuilder &builder, aiir::Location loc,
                           aiir::Type type, aiir::Value op1, aiir::Value op2,
                           bool isByRef) {
    genCombiner<OpType>(builder, loc, redId, type, op1, op2, isByRef);
  };

  return createDeclareReductionHelper<OpType>(converter, reductionOpName, type,
                                              loc, isByRef, genCombinerCB,
                                              genInitValueCB);
}

bool ReductionProcessor::doReductionByRef(aiir::Type reductionType) {
  if (forceByrefReduction)
    return true;

  if (!fir::isa_trivial(fir::unwrapRefType(reductionType)) &&
      !fir::isa_derived(fir::unwrapRefType(reductionType)))
    return true;

  return false;
}

bool ReductionProcessor::doReductionByRef(aiir::Value reductionVar) {
  if (forceByrefReduction)
    return true;

  if (auto declare =
          aiir::dyn_cast<hlfir::DeclareOp>(reductionVar.getDefiningOp()))
    reductionVar = declare.getMemref();

  return doReductionByRef(reductionVar.getType());
}

template <typename OpType, typename RedOperatorListTy>
bool ReductionProcessor::processReductionArguments(
    aiir::Location currentLocation, lower::AbstractConverter &converter,
    const RedOperatorListTy &redOperatorList,
    llvm::SmallVectorImpl<aiir::Value> &reductionVars,
    llvm::SmallVectorImpl<bool> &reduceVarByRef,
    llvm::SmallVectorImpl<aiir::Attribute> &reductionDeclSymbols,
    const llvm::SmallVectorImpl<const semantics::Symbol *> &reductionSymbols) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  if constexpr (std::is_same_v<RedOperatorListTy,
                               omp::clause::ReductionOperatorList>) {
    // For OpenMP reduction clauses, check if the reduction operator is
    // supported.
    assert(redOperatorList.size() == 1 && "Expecting single operator");
    const Fortran::lower::omp::clause::ReductionOperator &redOperator =
        redOperatorList.front();

    if (!std::holds_alternative<omp::clause::DefinedOperator>(redOperator.u)) {
      if (const auto *reductionIntrinsic =
              std::get_if<omp::clause::ProcedureDesignator>(&redOperator.u)) {
        if (!ReductionProcessor::supportedIntrinsicProcReduction(
                *reductionIntrinsic)) {
          // If not an intrinsic is has to be a custom reduction op, and should
          // be available in the module.
          semantics::Symbol *sym = reductionIntrinsic->v.sym();
          aiir::ModuleOp module = builder.getModule();
          auto decl = module.lookupSymbol<OpType>(getRealName(sym).ToString());
          if (!decl)
            return false;
        }
      } else {
        return false;
      }
    }
  }

  // Reduction variable processing common to both intrinsic operators and
  // procedure designators
  aiir::OpBuilder::InsertPoint dcIP;
  constexpr bool isDoConcurrent =
      std::is_same_v<OpType, fir::DeclareReductionOp>;

  if (isDoConcurrent) {
    dcIP = builder.saveInsertionPoint();
    builder.setInsertionPoint(
        builder.getRegion().getParentOfType<fir::DoConcurrentOp>());
  }

  for (const semantics::Symbol *symbol : reductionSymbols) {
    aiir::Value symVal = converter.getSymbolAddress(*symbol);

    if (auto declOp = symVal.getDefiningOp<hlfir::DeclareOp>())
      symVal = declOp.getBase();

    aiir::Type eleType;
    auto refType = aiir::dyn_cast_or_null<fir::ReferenceType>(symVal.getType());
    if (refType)
      eleType = refType.getEleTy();
    else
      eleType = symVal.getType();

    // all arrays must be boxed so that we have convenient access to all the
    // information needed to iterate over the array
    if (aiir::isa<fir::SequenceType>(eleType)) {
      // For Host associated symbols, use `SymbolBox` instead
      lower::SymbolBox symBox = converter.lookupOneLevelUpSymbol(*symbol);
      hlfir::Entity entity{symBox.getAddr()};
      entity = genVariableBox(currentLocation, builder, entity);
      aiir::Value box = entity.getBase();

      // Always pass the box by reference so that the OpenMP dialect
      // verifiers don't need to know anything about fir.box
      auto alloca =
          fir::AllocaOp::create(builder, currentLocation, box.getType());
      fir::StoreOp::create(builder, currentLocation, box, alloca);

      symVal = alloca;
    } else if (aiir::isa<fir::BaseBoxType>(symVal.getType())) {
      // boxed arrays are passed as values not by reference. Unfortunately,
      // we can't pass a box by value to omp.redution_declare, so turn it
      // into a reference
      auto oldIP = builder.saveInsertionPoint();
      builder.setInsertionPointToStart(builder.getAllocaBlock());
      auto alloca =
          fir::AllocaOp::create(builder, currentLocation, symVal.getType());
      builder.restoreInsertionPoint(oldIP);
      fir::StoreOp::create(builder, currentLocation, symVal, alloca);
      symVal = alloca;
    }

    // this isn't the same as the by-val and by-ref passing later in the
    // pipeline. Both styles assume that the variable is a reference at
    // this point
    assert(fir::isa_ref_type(symVal.getType()) &&
           "reduction input var is passed by reference");
    aiir::Type elementType = fir::dyn_cast_ptrEleTy(symVal.getType());
    const bool symIsVolatile = fir::isa_volatile_type(symVal.getType());
    aiir::Type refTy = fir::ReferenceType::get(elementType, symIsVolatile);

    reductionVars.push_back(
        builder.createConvert(currentLocation, refTy, symVal));
    reduceVarByRef.push_back(doReductionByRef(symVal));
  }

  unsigned idx = 0;
  for (auto [symVal, isByRef] : llvm::zip(reductionVars, reduceVarByRef)) {
    auto redType = aiir::cast<fir::ReferenceType>(symVal.getType());
    const auto &kindMap = builder.getKindMap();
    std::string reductionName;
    ReductionIdentifier redId;

    if constexpr (std::is_same_v<RedOperatorListTy,
                                 omp::clause::ReductionOperatorList>) {
      const Fortran::lower::omp::clause::ReductionOperator &redOperator =
          redOperatorList.front();
      if (const auto &redDefinedOp =
              std::get_if<omp::clause::DefinedOperator>(&redOperator.u)) {
        const auto &intrinsicOp{
            std::get<omp::clause::DefinedOperator::IntrinsicOperator>(
                redDefinedOp->u)};
        redId = getReductionType(intrinsicOp);
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

        reductionName = getReductionName(redId, kindMap, redType, isByRef);
      } else if (const auto *reductionIntrinsic =
                     std::get_if<omp::clause::ProcedureDesignator>(
                         &redOperator.u)) {
        if (!ReductionProcessor::supportedIntrinsicProcReduction(
                *reductionIntrinsic)) {
          // Custom reductions we can just add to the symbols without
          // generating the declare reduction op.
          semantics::Symbol *sym = reductionIntrinsic->v.sym();
          reductionDeclSymbols.push_back(aiir::SymbolRefAttr::get(
              builder.getContext(), sym->name().ToString()));
          ++idx;
          continue;
        }
        redId = getReductionType(*reductionIntrinsic);
        reductionName =
            getReductionName(getRealName(*reductionIntrinsic).ToString(),
                             kindMap, redType, isByRef);
      } else {
        TODO(currentLocation, "Unexpected reduction type");
      }
    } else {
      // `do concurrent` reductions
      redId = getReductionType(redOperatorList[idx]);
      reductionName = getReductionName(redId, kindMap, redType, isByRef);
    }

    OpType decl = createDeclareReduction<OpType>(
        converter, reductionName, redId, redType, currentLocation, isByRef);
    reductionDeclSymbols.push_back(
        aiir::SymbolRefAttr::get(builder.getContext(), decl.getSymName()));
    ++idx;
  }

  if (isDoConcurrent)
    builder.restoreInsertionPoint(dcIP);

  return true;
}

const semantics::SourceName
ReductionProcessor::getRealName(const semantics::Symbol *symbol) {
  return symbol->GetUltimate().name();
}

const semantics::SourceName
ReductionProcessor::getRealName(const omp::clause::ProcedureDesignator &pd) {
  return getRealName(pd.v.sym());
}

int ReductionProcessor::getOperationIdentity(ReductionIdentifier redId,
                                             aiir::Location loc) {
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
