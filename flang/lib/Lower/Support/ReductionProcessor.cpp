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
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
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
template void ReductionProcessor::processReductionArguments<
    mlir::omp::DeclareReductionOp, omp::clause::ReductionOperatorList>(
    mlir::Location currentLocation, lower::AbstractConverter &converter,
    const omp::clause::ReductionOperatorList &redOperatorList,
    llvm::SmallVectorImpl<mlir::Value> &reductionVars,
    llvm::SmallVectorImpl<bool> &reduceVarByRef,
    llvm::SmallVectorImpl<mlir::Attribute> &reductionDeclSymbols,
    const llvm::SmallVectorImpl<const semantics::Symbol *> &reductionSymbols);

template void ReductionProcessor::processReductionArguments<
    fir::DeclareReductionOp, llvm::SmallVector<fir::ReduceOperationEnum>>(
    mlir::Location currentLocation, lower::AbstractConverter &converter,
    const llvm::SmallVector<fir::ReduceOperationEnum> &redOperatorList,
    llvm::SmallVectorImpl<mlir::Value> &reductionVars,
    llvm::SmallVectorImpl<bool> &reduceVarByRef,
    llvm::SmallVectorImpl<mlir::Attribute> &reductionDeclSymbols,
    const llvm::SmallVectorImpl<const semantics::Symbol *> &reductionSymbols);

template mlir::omp::DeclareReductionOp
ReductionProcessor::createDeclareReduction<mlir::omp::DeclareReductionOp>(
    AbstractConverter &converter, llvm::StringRef reductionOpName,
    const ReductionIdentifier redId, mlir::Type type, mlir::Location loc,
    bool isByRef);

template fir::DeclareReductionOp
ReductionProcessor::createDeclareReduction<fir::DeclareReductionOp>(
    AbstractConverter &converter, llvm::StringRef reductionOpName,
    const ReductionIdentifier redId, mlir::Type type, mlir::Location loc,
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
                                     mlir::Type ty, bool isByRef) {
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
                                     mlir::Type ty, bool isByRef) {
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

mlir::Value
ReductionProcessor::getReductionInitValue(mlir::Location loc, mlir::Type type,
                                          ReductionIdentifier redId,
                                          fir::FirOpBuilder &builder) {
  type = fir::unwrapRefType(type);
  if (!fir::isa_integer(type) && !fir::isa_real(type) &&
      !fir::isa_complex(type) && !mlir::isa<fir::LogicalType>(type))
    TODO(loc, "Reduction of some types is not supported");
  switch (redId) {
  case ReductionIdentifier::MAX: {
    if (auto ty = mlir::dyn_cast<mlir::FloatType>(type)) {
      const llvm::fltSemantics &sem = ty.getFloatSemantics();
      return builder.createRealConstant(
          loc, type, llvm::APFloat::getLargest(sem, /*Negative=*/true));
    }
    unsigned bits = type.getIntOrFloatBitWidth();
    int64_t minInt = llvm::APInt::getSignedMinValue(bits).getSExtValue();
    return builder.createIntegerConstant(loc, type, minInt);
  }
  case ReductionIdentifier::MIN: {
    if (auto ty = mlir::dyn_cast<mlir::FloatType>(type)) {
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
    if (auto cplxTy = mlir::dyn_cast<mlir::ComplexType>(type)) {
      mlir::Type realTy = cplxTy.getElementType();
      mlir::Value initRe = builder.createRealConstant(
          loc, realTy, getOperationIdentity(redId, loc));
      mlir::Value initIm = builder.createRealConstant(loc, realTy, 0);

      return fir::factory::Complex{builder, loc}.createComplex(type, initRe,
                                                               initIm);
    }
    if (mlir::isa<mlir::FloatType>(type))
      return mlir::arith::ConstantOp::create(
          builder, loc, type,
          builder.getFloatAttr(type, (double)getOperationIdentity(redId, loc)));

    if (mlir::isa<fir::LogicalType>(type)) {
      mlir::Value intConst = mlir::arith::ConstantOp::create(
          builder, loc, builder.getI1Type(),
          builder.getIntegerAttr(builder.getI1Type(),
                                 getOperationIdentity(redId, loc)));
      return builder.createConvert(loc, type, intConst);
    }

    return mlir::arith::ConstantOp::create(
        builder, loc, type,
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
        getReductionOperation<mlir::arith::MaxNumFOp, mlir::arith::MaxSIOp>(
            builder, type, loc, op1, op2);
    break;
  case ReductionIdentifier::MIN:
    reductionOp =
        getReductionOperation<mlir::arith::MinNumFOp, mlir::arith::MinSIOp>(
            builder, type, loc, op1, op2);
    break;
  case ReductionIdentifier::IOR:
    assert((type.isIntOrIndex()) && "only integer is expected");
    reductionOp = mlir::arith::OrIOp::create(builder, loc, op1, op2);
    break;
  case ReductionIdentifier::IEOR:
    assert((type.isIntOrIndex()) && "only integer is expected");
    reductionOp = mlir::arith::XOrIOp::create(builder, loc, op1, op2);
    break;
  case ReductionIdentifier::IAND:
    assert((type.isIntOrIndex()) && "only integer is expected");
    reductionOp = mlir::arith::AndIOp::create(builder, loc, op1, op2);
    break;
  case ReductionIdentifier::ADD:
    reductionOp =
        getReductionOperation<mlir::arith::AddFOp, mlir::arith::AddIOp,
                              fir::AddcOp>(builder, type, loc, op1, op2);
    break;
  case ReductionIdentifier::MULTIPLY:
    reductionOp =
        getReductionOperation<mlir::arith::MulFOp, mlir::arith::MulIOp,
                              fir::MulcOp>(builder, type, loc, op1, op2);
    break;
  case ReductionIdentifier::AND: {
    mlir::Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    mlir::Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    mlir::Value andiOp =
        mlir::arith::AndIOp::create(builder, loc, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, andiOp);
    break;
  }
  case ReductionIdentifier::OR: {
    mlir::Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    mlir::Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    mlir::Value oriOp = mlir::arith::OrIOp::create(builder, loc, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, oriOp);
    break;
  }
  case ReductionIdentifier::EQV: {
    mlir::Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    mlir::Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    mlir::Value cmpiOp = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::eq, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, cmpiOp);
    break;
  }
  case ReductionIdentifier::NEQV: {
    mlir::Value op1I1 = builder.createConvert(loc, builder.getI1Type(), op1);
    mlir::Value op2I1 = builder.createConvert(loc, builder.getI1Type(), op2);

    mlir::Value cmpiOp = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::ne, op1I1, op2I1);

    reductionOp = builder.createConvert(loc, type, cmpiOp);
    break;
  }
  default:
    TODO(loc, "Reduction of some intrinsic operators is not supported");
  }

  return reductionOp;
}

template <typename ParentDeclOpType>
static void genYield(fir::FirOpBuilder &builder, mlir::Location loc,
                     mlir::Value yieldedValue) {
  if constexpr (std::is_same_v<ParentDeclOpType, mlir::omp::DeclareReductionOp>)
    mlir::omp::YieldOp::create(builder, loc, yieldedValue);
  else
    fir::YieldOp::create(builder, loc, yieldedValue);
}

/// Create reduction combiner region for reduction variables which are boxed
/// arrays
template <typename DeclRedOpType>
static void genBoxCombiner(fir::FirOpBuilder &builder, mlir::Location loc,
                           ReductionProcessor::ReductionIdentifier redId,
                           fir::BaseBoxType boxTy, mlir::Value lhs,
                           mlir::Value rhs) {
  fir::SequenceType seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(
      fir::unwrapRefType(boxTy.getEleTy()));
  fir::HeapType heapTy =
      mlir::dyn_cast_or_null<fir::HeapType>(boxTy.getEleTy());
  fir::PointerType ptrTy =
      mlir::dyn_cast_or_null<fir::PointerType>(boxTy.getEleTy());
  if ((!seqTy || seqTy.hasUnknownShape()) && !heapTy && !ptrTy)
    TODO(loc, "Unsupported boxed type in OpenMP reduction");

  // load fir.ref<fir.box<...>>
  mlir::Value lhsAddr = lhs;
  lhs = fir::LoadOp::create(builder, loc, lhs);
  rhs = fir::LoadOp::create(builder, loc, rhs);

  if ((heapTy || ptrTy) && !seqTy) {
    // get box contents (heap pointers)
    lhs = fir::BoxAddrOp::create(builder, loc, lhs);
    rhs = fir::BoxAddrOp::create(builder, loc, rhs);
    mlir::Value lhsValAddr = lhs;

    // load heap pointers
    lhs = fir::LoadOp::create(builder, loc, lhs);
    rhs = fir::LoadOp::create(builder, loc, rhs);

    mlir::Type eleTy = heapTy ? heapTy.getEleTy() : ptrTy.getEleTy();

    mlir::Value result = ReductionProcessor::createScalarCombiner(
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
  mlir::Type refTy = fir::ReferenceType::get(seqTy.getEleTy(), seqIsVolatile);
  auto lhsEleAddr = fir::ArrayCoorOp::create(
      builder, loc, refTy, lhs, shapeShift, /*slice=*/mlir::Value{},
      nest.oneBasedIndices, /*typeparms=*/mlir::ValueRange{});
  auto rhsEleAddr = fir::ArrayCoorOp::create(
      builder, loc, refTy, rhs, shapeShift, /*slice=*/mlir::Value{},
      nest.oneBasedIndices, /*typeparms=*/mlir::ValueRange{});
  auto lhsEle = fir::LoadOp::create(builder, loc, lhsEleAddr);
  auto rhsEle = fir::LoadOp::create(builder, loc, rhsEleAddr);
  mlir::Value scalarReduction = ReductionProcessor::createScalarCombiner(
      builder, loc, redId, refTy, lhsEle, rhsEle);
  fir::StoreOp::create(builder, loc, scalarReduction, lhsEleAddr);

  builder.setInsertionPointAfter(nest.outerOp);
  genYield<DeclRedOpType>(builder, loc, lhsAddr);
}

// generate combiner region for reduction operations
template <typename DeclRedOpType>
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
      fir::StoreOp::create(builder, loc, result, lhs);
      genYield<DeclRedOpType>(builder, loc, lhs);
    } else {
      genYield<DeclRedOpType>(builder, loc, result);
    }
    return;
  }
  // all arrays should have been boxed
  if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(ty)) {
    genBoxCombiner<DeclRedOpType>(builder, loc, redId, boxTy, lhs, rhs);
    return;
  }

  TODO(loc, "OpenMP genCombiner for unsupported reduction variable type");
}

// like fir::unwrapSeqOrBoxedSeqType except it also works for non-sequence boxes
static mlir::Type unwrapSeqOrBoxedType(mlir::Type ty) {
  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(ty))
    return seqTy.getEleTy();
  if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(ty)) {
    auto eleTy = fir::unwrapRefType(boxTy.getEleTy());
    if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(eleTy))
      return seqTy.getEleTy();
    return eleTy;
  }
  return ty;
}

template <typename OpType>
static void createReductionAllocAndInitRegions(
    AbstractConverter &converter, mlir::Location loc, OpType &reductionDecl,
    const ReductionProcessor::ReductionIdentifier redId, mlir::Type type,
    bool isByRef) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  auto yield = [&](mlir::Value ret) { genYield<OpType>(builder, loc, ret); };

  mlir::Block *allocBlock = nullptr;
  mlir::Block *initBlock = nullptr;
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

  mlir::Type ty = fir::unwrapRefType(type);
  builder.setInsertionPointToEnd(initBlock);
  mlir::Value initValue = ReductionProcessor::getReductionInitValue(
      loc, unwrapSeqOrBoxedType(ty), redId, builder);

  if (isByRef) {
    populateByRefInitAndCleanupRegions(
        converter, loc, type, initValue, initBlock,
        reductionDecl.getInitializerAllocArg(),
        reductionDecl.getInitializerMoldArg(), reductionDecl.getCleanupRegion(),
        DeclOperationKind::Reduction, /*sym=*/nullptr,
        /*cannotHaveLowerBounds=*/false,
        /*isDoConcurrent*/ std::is_same_v<OpType, fir::DeclareReductionOp>);
  }

  if (fir::isa_trivial(ty)) {
    if (isByRef) {
      // alloc region
      builder.setInsertionPointToEnd(allocBlock);
      mlir::Value alloca = fir::AllocaOp::create(builder, loc, ty);
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
  mlir::Value boxAlloca = fir::AllocaOp::create(builder, loc, ty);
  yield(boxAlloca);
}

template <typename OpType>
OpType ReductionProcessor::createDeclareReduction(
    AbstractConverter &converter, llvm::StringRef reductionOpName,
    const ReductionIdentifier redId, mlir::Type type, mlir::Location loc,
    bool isByRef) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::ModuleOp module = builder.getModule();

  assert(!reductionOpName.empty());

  auto decl = module.lookupSymbol<OpType>(reductionOpName);
  if (decl)
    return decl;

  mlir::OpBuilder modBuilder(module.getBodyRegion());
  mlir::Type valTy = fir::unwrapRefType(type);
  if (!isByRef)
    type = valTy;

  decl = OpType::create(modBuilder, loc, reductionOpName, type);
  createReductionAllocAndInitRegions(converter, loc, decl, redId, type,
                                     isByRef);

  builder.createBlock(&decl.getReductionRegion(),
                      decl.getReductionRegion().end(), {type, type},
                      {loc, loc});

  builder.setInsertionPointToEnd(&decl.getReductionRegion().back());
  mlir::Value op1 = decl.getReductionRegion().front().getArgument(0);
  mlir::Value op2 = decl.getReductionRegion().front().getArgument(1);
  genCombiner<OpType>(builder, loc, redId, type, op1, op2, isByRef);

  return decl;
}

static bool doReductionByRef(mlir::Value reductionVar) {
  if (forceByrefReduction)
    return true;

  if (auto declare =
          mlir::dyn_cast<hlfir::DeclareOp>(reductionVar.getDefiningOp()))
    reductionVar = declare.getMemref();

  if (!fir::isa_trivial(fir::unwrapRefType(reductionVar.getType())))
    return true;

  return false;
}

template <typename OpType, typename RedOperatorListTy>
void ReductionProcessor::processReductionArguments(
    mlir::Location currentLocation, lower::AbstractConverter &converter,
    const RedOperatorListTy &redOperatorList,
    llvm::SmallVectorImpl<mlir::Value> &reductionVars,
    llvm::SmallVectorImpl<bool> &reduceVarByRef,
    llvm::SmallVectorImpl<mlir::Attribute> &reductionDeclSymbols,
    const llvm::SmallVectorImpl<const semantics::Symbol *> &reductionSymbols) {
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
          return;
        }
      } else {
        return;
      }
    }
  }

  // Reduction variable processing common to both intrinsic operators and
  // procedure designators
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::OpBuilder::InsertPoint dcIP;
  constexpr bool isDoConcurrent =
      std::is_same_v<OpType, fir::DeclareReductionOp>;

  if (isDoConcurrent) {
    dcIP = builder.saveInsertionPoint();
    builder.setInsertionPoint(
        builder.getRegion().getParentOfType<fir::DoConcurrentOp>());
  }

  for (const semantics::Symbol *symbol : reductionSymbols) {
    mlir::Value symVal = converter.getSymbolAddress(*symbol);

    if (auto declOp = symVal.getDefiningOp<hlfir::DeclareOp>())
      symVal = declOp.getBase();

    mlir::Type eleType;
    auto refType = mlir::dyn_cast_or_null<fir::ReferenceType>(symVal.getType());
    if (refType)
      eleType = refType.getEleTy();
    else
      eleType = symVal.getType();

    // all arrays must be boxed so that we have convenient access to all the
    // information needed to iterate over the array
    if (mlir::isa<fir::SequenceType>(eleType)) {
      // For Host associated symbols, use `SymbolBox` instead
      lower::SymbolBox symBox = converter.lookupOneLevelUpSymbol(*symbol);
      hlfir::Entity entity{symBox.getAddr()};
      entity = genVariableBox(currentLocation, builder, entity);
      mlir::Value box = entity.getBase();

      // Always pass the box by reference so that the OpenMP dialect
      // verifiers don't need to know anything about fir.box
      auto alloca =
          fir::AllocaOp::create(builder, currentLocation, box.getType());
      fir::StoreOp::create(builder, currentLocation, box, alloca);

      symVal = alloca;
    } else if (mlir::isa<fir::BaseBoxType>(symVal.getType())) {
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
    mlir::Type elementType = fir::dyn_cast_ptrEleTy(symVal.getType());
    const bool symIsVolatile = fir::isa_volatile_type(symVal.getType());
    mlir::Type refTy = fir::ReferenceType::get(elementType, symIsVolatile);

    reductionVars.push_back(
        builder.createConvert(currentLocation, refTy, symVal));
    reduceVarByRef.push_back(doReductionByRef(symVal));
  }

  unsigned idx = 0;
  for (auto [symVal, isByRef] : llvm::zip(reductionVars, reduceVarByRef)) {
    auto redType = mlir::cast<fir::ReferenceType>(symVal.getType());
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
          TODO(currentLocation, "Unsupported intrinsic proc reduction");
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
        mlir::SymbolRefAttr::get(builder.getContext(), decl.getSymName()));
    ++idx;
  }

  if (isDoConcurrent)
    builder.restoreInsertionPoint(dcIP);
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
