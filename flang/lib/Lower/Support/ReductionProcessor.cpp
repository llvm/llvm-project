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
#include "flang/Lower/ConvertExprToHLFIR.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/OpenMP.h"
#include "flang/Lower/OpenMP/Clauses.h"
#include "flang/Lower/Support/PrivateReductionUtils.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Semantics/openmp-utils.h"
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
template bool ReductionProcessor::processReductionArguments<
    mlir::omp::DeclareReductionOp, omp::clause::ReductionOperatorList>(
    mlir::Location currentLocation, lower::AbstractConverter &converter,
    const omp::clause::ReductionOperatorList &redOperatorList,
    llvm::SmallVectorImpl<mlir::Value> &reductionVars,
    llvm::SmallVectorImpl<bool> &reduceVarByRef,
    llvm::SmallVectorImpl<mlir::Attribute> &reductionDeclSymbols,
    const llvm::SmallVectorImpl<const semantics::Symbol *> &reductionSymbols,
    llvm::ArrayRef<Object> reductionObjects, lower::SymMap &symMap,
    semantics::SemanticsContext *semaCtx,
    llvm::DenseMap<const semantics::Symbol *, mlir::Value> *reductionVarCache);

template bool ReductionProcessor::processReductionArguments<
    fir::DeclareReductionOp, llvm::SmallVector<fir::ReduceOperationEnum>>(
    mlir::Location currentLocation, lower::AbstractConverter &converter,
    const llvm::SmallVector<fir::ReduceOperationEnum> &redOperatorList,
    llvm::SmallVectorImpl<mlir::Value> &reductionVars,
    llvm::SmallVectorImpl<bool> &reduceVarByRef,
    llvm::SmallVectorImpl<mlir::Attribute> &reductionDeclSymbols,
    const llvm::SmallVectorImpl<const semantics::Symbol *> &reductionSymbols,
    llvm::ArrayRef<Object> reductionObjects, lower::SymMap &symMap,
    semantics::SemanticsContext *semaCtx,
    llvm::DenseMap<const semantics::Symbol *, mlir::Value> *reductionVarCache);

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

// Bridge the clause-level intrinsic operator to the parser-level enum consumed
// by semantics' MakeNameFromOperator, so the mangled reduction name ("op.+",
// "op.AND", ...) is produced by the single semantics source of truth rather
// than re-spelled here. Only the operators a user reduction may use reach this
// (the intrinsic-operator clause path filters the rest with a TODO first).
parser::DefinedOperator::IntrinsicOperator
ReductionProcessor::toParserIntrinsicOperator(
    omp::clause::DefinedOperator::IntrinsicOperator op) {
  using C = omp::clause::DefinedOperator::IntrinsicOperator;
  using P = parser::DefinedOperator::IntrinsicOperator;
  switch (op) {
  case C::Add:
    return P::Add;
  case C::Multiply:
    return P::Multiply;
  case C::AND:
    return P::AND;
  case C::OR:
    return P::OR;
  case C::EQV:
    return P::EQV;
  case C::NEQV:
    return P::NEQV;
  default:
    llvm_unreachable("unexpected intrinsic operator in user reduction");
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
  case ReductionIdentifier::MAX:
    reductionName = "max_reduction";
    break;
  case ReductionIdentifier::MIN:
    reductionName = "min_reduction";
    break;
  case ReductionIdentifier::IAND:
    reductionName = "iand_reduction";
    break;
  case ReductionIdentifier::IOR:
    reductionName = "ior_reduction";
    break;
  case ReductionIdentifier::IEOR:
    reductionName = "ieor_reduction";
    break;
  default:
    llvm_unreachable("unsupported reduction identifier");
  }

  return getReductionName(reductionName, kindMap, ty, isByRef);
}

std::string ReductionProcessor::getScopedUserReductionName(
    AbstractConverter &converter, const semantics::Symbol &reductionSymbol,
    mlir::Type reductionType, bool isByRef) {
  // Qualify the reduction symbol's ultimate name with its owning scope so that
  // user-defined reductions with the same spelling in different modules get
  // distinct op names. Use the (name, scope) mangleName overload: the
  // (symbol) overload does not handle UserReductionDetails.
  const semantics::Symbol &ultimate = reductionSymbol.GetUltimate();
  std::string name = ultimate.name().ToString();
  std::string scopedName = converter.mangleName(name, ultimate.owner());
  // Append the type and by-ref suffix, as the intrinsic reductions do, so a
  // declare reduction listing several types produces one op per type and every
  // name ends in a type-grammar token. Suffix unconditionally, even for one
  // type: otherwise a multi-type myred's per-type op "myred_i32" would collide
  // with a single-type reduction named "myred_i32", and the by-name dedup would
  // bind one reduction's clause to the other's combiner (a silent miscompile).
  // The by-ref token also gives an allocatable/pointer trivial reduction (a
  // boxed by-ref operand of a by-value declared type) a name the directive
  // never emitted, so it reaches a clean TODO instead of binding a by-value op
  // to a box (handled by llvm-project#186765).
  return getReductionName(scopedName, converter.getFirOpBuilder().getKindMap(),
                          reductionType, isByRef);
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
    mlir::Value v1 = builder.createConvert(loc, type, op1);
    mlir::Value v2 = builder.createConvert(loc, type, op2);
    reductionOp = fir::LogicalAndOp::create(builder, loc, type, v1, v2);
    break;
  }
  case ReductionIdentifier::OR: {
    mlir::Value v1 = builder.createConvert(loc, type, op1);
    mlir::Value v2 = builder.createConvert(loc, type, op2);
    reductionOp = fir::LogicalOrOp::create(builder, loc, type, v1, v2);
    break;
  }
  case ReductionIdentifier::EQV: {
    mlir::Value v1 = builder.createConvert(loc, type, op1);
    mlir::Value v2 = builder.createConvert(loc, type, op2);
    reductionOp = fir::EqvOp::create(builder, loc, type, v1, v2);
    break;
  }
  case ReductionIdentifier::NEQV: {
    mlir::Value v1 = builder.createConvert(loc, type, op1);
    mlir::Value v2 = builder.createConvert(loc, type, op2);
    reductionOp = fir::NeqvOp::create(builder, loc, type, v1, v2);
    break;
  }
  default:
    TODO(loc, "Reduction of some intrinsic operators is not supported");
  }

  return reductionOp;
}

bool ReductionProcessor::isExpressionLoweredAsReductionObject(
    const Object *object) {
  if (!object || !object->ref())
    return false;
  const SomeExpr &expr = *object->ref();
  return evaluate::IsArrayElement(expr);
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
    ReductionProcessor::GenInitValueCBTy genInitValueCB, mlir::Type type,
    bool isByRef, const Fortran::semantics::Symbol *sym) {
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
  mlir::Value initValue =
      isByRef ? genInitValueCB(builder, loc, ty, initBlock->getArgument(0),
                               initBlock->getArgument(1))
              : genInitValueCB(builder, loc, ty, initBlock->getArgument(0),
                               mlir::Value{});
  if (isByRef) {
    populateByRefInitAndCleanupRegions(
        converter, loc, type, initValue, initBlock,
        reductionDecl.getInitializerAllocArg(),
        reductionDecl.getInitializerMoldArg(), reductionDecl.getCleanupRegion(),
        DeclOperationKind::Reduction, sym,
        /*cannotHaveLowerBounds=*/false,
        /*isDoConcurrent*/ std::is_same_v<OpType, fir::DeclareReductionOp>);
  }

  if (fir::isa_trivial(ty) || fir::isa_derived(ty)) {
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

template <typename DeclareRedType>
DeclareRedType ReductionProcessor::createDeclareReductionHelper(
    AbstractConverter &converter, llvm::StringRef reductionOpName,
    mlir::Type type, mlir::Location loc, bool isByRef,
    GenCombinerCBTy genCombinerCB, GenInitValueCBTy genInitValueCB,
    const semantics::Symbol *sym) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::ModuleOp module = builder.getModule();

  assert(!reductionOpName.empty());

  auto decl = module.lookupSymbol<DeclareRedType>(reductionOpName);
  if (decl)
    return decl;

  mlir::OpBuilder modBuilder(module.getBodyRegion());
  mlir::Type valTy = fir::unwrapRefType(type);

  // For by-ref reductions, we want to keep track of the
  // boxed/referenced/allocated type. For example, for a `real, allocatable`
  // variable, `real` should be stored.
  mlir::TypeAttr boxedTyAttr{};
  mlir::Type boxedTy;

  if (isByRef) {
    boxedTy = fir::unwrapPassByRefType(valTy);
    boxedTyAttr = mlir::TypeAttr::get(boxedTy);
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
                                     isByRef, sym);
  builder.createBlock(&decl.getReductionRegion(),
                      decl.getReductionRegion().end(), {type, type},
                      {loc, loc});
  builder.setInsertionPointToEnd(&decl.getReductionRegion().back());
  mlir::Value op1 = decl.getReductionRegion().front().getArgument(0);
  mlir::Value op2 = decl.getReductionRegion().front().getArgument(1);
  genCombinerCB(builder, loc, type, op1, op2, isByRef);

  if (isByRef && fir::isa_box_type(valTy)) {
    mlir::Region &dataPtrPtrRegion = decl.getDataPtrPtrRegion();
    mlir::Block &dataAddrBlock = *builder.createBlock(
        &dataPtrPtrRegion, dataPtrPtrRegion.end(), {type}, {loc});
    builder.setInsertionPointToEnd(&dataAddrBlock);
    mlir::Value boxRefOperand = dataAddrBlock.getArgument(0);
    mlir::Value baseAddrOffset = fir::BoxOffsetOp::create(
        builder, loc, boxRefOperand, fir::BoxFieldAttr::base_addr);
    genYield<DeclareRedType>(builder, loc, baseAddrOffset);
  }

  return decl;
}

template <typename OpType>
OpType ReductionProcessor::createDeclareReduction(
    AbstractConverter &converter, llvm::StringRef reductionOpName,
    const ReductionIdentifier redId, mlir::Type type, mlir::Location loc,
    bool isByRef) {
  auto genInitValueCB = [&](fir::FirOpBuilder &builder, mlir::Location loc,
                            mlir::Type type, mlir::Value /*moldArg*/,
                            mlir::Value /*privArg*/) {
    mlir::Type ty = fir::unwrapRefType(type);
    mlir::Value initValue = ReductionProcessor::getReductionInitValue(
        loc, unwrapSeqOrBoxedType(ty), redId, builder);
    return initValue;
  };
  auto genCombinerCB = [&](fir::FirOpBuilder &builder, mlir::Location loc,
                           mlir::Type type, mlir::Value op1, mlir::Value op2,
                           bool isByRef) {
    genCombiner<OpType>(builder, loc, redId, type, op1, op2, isByRef);
  };

  return createDeclareReductionHelper<OpType>(converter, reductionOpName, type,
                                              loc, isByRef, genCombinerCB,
                                              genInitValueCB);
}

bool ReductionProcessor::doReductionByRef(mlir::Type reductionType) {
  if (forceByrefReduction)
    return true;
  // Non-trivial, non-derived types (e.g., boxes, arrays) must be by-ref.
  // Derived types must also be by-ref because user-defined combiners
  // operate on components via side-effects, not by producing a whole value.
  if (!fir::isa_trivial(fir::unwrapRefType(reductionType)))
    return true;
  return false;
}

bool ReductionProcessor::doReductionByRef(mlir::Value reductionVar) {
  if (forceByrefReduction)
    return true;

  if (auto declare =
          mlir::dyn_cast<hlfir::DeclareOp>(reductionVar.getDefiningOp()))
    reductionVar = declare.getMemref();

  return doReductionByRef(reductionVar.getType());
}

template <typename OpType, typename RedOperatorListTy>
bool ReductionProcessor::processReductionArguments(
    mlir::Location currentLocation, lower::AbstractConverter &converter,
    const RedOperatorListTy &redOperatorList,
    llvm::SmallVectorImpl<mlir::Value> &reductionVars,
    llvm::SmallVectorImpl<bool> &reduceVarByRef,
    llvm::SmallVectorImpl<mlir::Attribute> &reductionDeclSymbols,
    const llvm::SmallVectorImpl<const semantics::Symbol *> &reductionSymbols,
    llvm::ArrayRef<Object> reductionObjects, lower::SymMap &symMap,
    semantics::SemanticsContext *semaCtx,
    llvm::DenseMap<const semantics::Symbol *, mlir::Value> *reductionVarCache) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  if constexpr (std::is_same_v<RedOperatorListTy,
                               omp::clause::ReductionOperatorList>) {
    // For OpenMP reduction clauses, check if the reduction operator is
    // supported.
    assert(redOperatorList.size() == 1 && "Expecting single operator");
    const Fortran::lower::omp::clause::ReductionOperator &redOperator =
        redOperatorList.front();

    if (!std::holds_alternative<omp::clause::DefinedOperator>(redOperator.u)) {
      // A named (procedure-designator) reduction, the only other alternative.
      // Defer validation to the per-variable loop below: a named reduction's op
      // is named per the variable's type (getScopedUserReductionName appends
      // the type suffix), so its existence and type can only be checked once
      // the variable type is known. The loop resolves the symbol, confirms its
      // ultimate has UserReductionDetails, looks the op up by the type-specific
      // name, and emits a clean TODO if missing or type-mismatched.
      assert(std::holds_alternative<omp::clause::ProcedureDesignator>(
                 redOperator.u) &&
             "ReductionIdentifier variant has only DefinedOperator and "
             "ProcedureDesignator");
    }
  }

  // Reduction variable processing common to both intrinsic operators and
  // procedure designators
  mlir::OpBuilder::InsertPoint dcIP;
  constexpr bool isDoConcurrent =
      std::is_same_v<OpType, fir::DeclareReductionOp>;

  if (isDoConcurrent) {
    dcIP = builder.saveInsertionPoint();
    builder.setInsertionPoint(
        builder.getRegion().getParentOfType<fir::DoConcurrentOp>());
  }

  assert((reductionObjects.empty() ||
          reductionSymbols.size() == reductionObjects.size()) &&
         "mismatched reduction symbol and object lists");

  for (unsigned i = 0; i < reductionSymbols.size(); ++i) {
    const Object *object =
        reductionObjects.empty() ? nullptr : &reductionObjects[i];
    const semantics::Symbol *symbol =
        object ? object->sym() : reductionSymbols[i];
    const SomeExpr *expr = object && object->ref() ? &*object->ref() : nullptr;
    const bool isObjectExpr =
        ReductionProcessor::isExpressionLoweredAsReductionObject(object);

    // If a cached reduction variable exists for this symbol, reuse it.
    // This ensures that composite constructs (e.g. DO SIMD) where both
    // the outer wrapper (wsloop) and inner wrapper (simd) process the same
    // reduction clause share the same SSA value, enabling genLoopVars()'s
    // IRMapping to correctly remap inner wrapper operands to outer wrapper
    // block arguments. Array element reductions are intentionally not cached:
    // block-argument object tracking maps their scoped uses.
    if (reductionVarCache && !isObjectExpr) {
      if (auto it = reductionVarCache->find(symbol);
          it != reductionVarCache->end()) {
        reductionVars.push_back(it->second);
        reduceVarByRef.push_back(doReductionByRef(it->second));
        continue;
      }
    }

    mlir::Value reductionVal;
    mlir::Type refTy;

    if (isObjectExpr) {
      StatementContext stmtCtx;
      hlfir::EntityWithAttributes entity = convertExprToHLFIR(
          converter.getCurrentLocation(), converter, *expr, symMap, stmtCtx);
      reductionVal = entity.getBase();
      // TODO Add support for Boxed and Sequenced types once these are supported
      refTy = reductionVal.getType();
    } else {
      mlir::Value symVal = converter.getSymbolAddress(*symbol);

      if (auto declOp = symVal.getDefiningOp<hlfir::DeclareOp>())
        symVal = declOp.getBase();

      mlir::Type eleType;
      auto refType =
          mlir::dyn_cast_or_null<fir::ReferenceType>(symVal.getType());
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
      refTy = fir::ReferenceType::get(elementType, symIsVolatile);
      reductionVal = symVal;
    }
    reductionVars.push_back(
        builder.createConvert(currentLocation, refTy, reductionVal));
    reduceVarByRef.push_back(doReductionByRef(reductionVars.back()));

    // Cache the final SSA value for this symbol so that subsequent calls
    // (e.g. for the inner wrapper in a composite construct) reuse it.
    if (reductionVarCache && !isObjectExpr)
      reductionVarCache->try_emplace(symbol, reductionVars.back());
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
      // Name user-defined reduction ops from the canonical element type, not
      // the raw lowered variable type. An allocatable/pointer variable lowers
      // to a boxed reference (!fir.ref<!fir.box<!fir.heap<!fir.char<1>>>>), but
      // the directive names its op from the declared element type
      // (!fir.char<1>); getReductionName unwraps only one reference level, so
      // without stripping the ref/box/heap/pointer wrappers the by-ref suffix
      // diverges and a valid allocatable-character reduction is wrongly
      // rejected. Keep any array fir::SequenceType. Only naming and the type
      // check use namingType; redType stays intact for op binding.
      mlir::Type namingType =
          fir::unwrapPassByRefType(fir::unwrapRefType(redType));
      // An allocatable/pointer reduction of a trivial element type is a case
      // the directive does not materialize: it emits the scalar by-value op,
      // and binding the boxed by-ref operand to it is invalid IR. In the
      // default mode the by-ref name suffix diverges (boxed clause by-ref,
      // scalar op by-value), the lookup misses, and it is a clean TODO. Under
      // -mmlir
      // --force-byref-reduction both sides are forced by-ref, the names match,
      // and the element-only type check (i32 == i32) would bind the box to the
      // scalar op. Guard on the inherent triviality of the element type, not
      // doReductionByRef (which the flag forces), so it is a clean TODO in
      // every mode; boxed character and derived reductions (genuinely by-ref)
      // still bind. Deferred to flang PR #186765.
      const bool isBoxedTrivialReduction =
          mlir::isa<fir::BaseBoxType>(fir::unwrapRefType(redType)) &&
          fir::isa_trivial(namingType);
      if (const auto &redDefinedOp =
              std::get_if<omp::clause::DefinedOperator>(&redOperator.u)) {
        if (const auto *definedOpName =
                std::get_if<omp::clause::DefinedOperator::DefinedOpName>(
                    &redDefinedOp->u)) {
          // User-defined operator reduction (e.g. reduction(.myop.:x)). Resolve
          // the use-site operator to the source reduction symbol as the OpenMP
          // semantic checks do (resolving from the clause scope, so it follows
          // USE and host association, operator renames, private visibility, and
          // merged generics), then reference the
          // omp.declare_reduction op named from that symbol's scoped (name,
          // owner). Reductions in different source modules have distinct owner
          // scopes, so a same-spelling/same-type operator reduction imported
          // from two modules yields two distinct ops, each clause binding its
          // own combiner instead of colliding onto one. The variable's type
          // selects the matching per-type op name (getScopedUserReductionName
          // appends the type), so a multiple-declaration/multiple-type operator
          // is handled (one op per type). An operator with no reduction for the
          // variable's type, or a variable with no type, is a clean TODO rather
          // than a crash or a wrong binding.
          const semantics::Symbol *opSym = definedOpName->v.sym();
          const semantics::DeclTypeSpec *varType =
              reductionSymbols[idx]->GetUltimate().GetType();
          // The variable's type disambiguates an operator that carries
          // reductions for several types (e.g. a generic merged from multiple
          // single-type modules): the resolver returns only a reduction that
          // supports varType.
          const semantics::Symbol *resolvedSym =
              semantics::omp::FindOperatorUserReductionSymbol(
                  converter.getCurrentScope(), *opSym, varType);
          // resolvedSym is the found symbol, which may be a USE-associated
          // wrapper; take its ultimate before reading the reduction details.
          const semantics::Symbol *ultimate =
              resolvedSym ? &resolvedSym->GetUltimate() : nullptr;
          const semantics::UserReductionDetails *userDetails =
              ultimate ? ultimate->detailsIf<semantics::UserReductionDetails>()
                       : nullptr;
          if (!varType || !resolvedSym || !userDetails) {
            TODO(currentLocation,
                 "OpenMP user-defined operator reduction is not yet supported "
                 "when the variable has no type or no matching reduction is "
                 "visible for its type");
          }
          std::string opName = ReductionProcessor::getScopedUserReductionName(
              converter, *resolvedSym, namingType, isByRef);
          mlir::ModuleOp module = builder.getModule();
          auto existingDecl = module.lookupSymbol<OpType>(opName);
          // Separate compilation: the primary pass lowers only this TU's own
          // declare reductions, so an imported reduction's op is absent here.
          // Materialize it on demand from the resolved symbol when its defining
          // module is a mod file, then re-look it up. A same-file op already
          // exists, so the ModFile gate keeps this separate-compilation only.
          if (!existingDecl && semaCtx && ultimate->owner().symbol() &&
              ultimate->owner().symbol()->test(
                  semantics::Symbol::Flag::ModFile)) {
            Fortran::lower::materializeUserReduction(
                converter, *semaCtx, *ultimate, opName, namingType, isByRef);
            existingDecl = module.lookupSymbol<OpType>(opName);
          }
          // The MLIR verifier does not type-check these ops (they have no
          // atomic region), so this is the only guard against binding a
          // mismatched declaration. Compare unwrapped element types: namingType
          // is the canonical reduction element type (allocatable/pointer
          // storage wrappers stripped), matching the type the op was named and
          // created from on the directive side.
          if (!existingDecl ||
              fir::unwrapRefType(existingDecl.getType()) !=
                  fir::unwrapRefType(namingType) ||
              isBoxedTrivialReduction) {
            TODO(currentLocation,
                 "OpenMP user-defined operator reduction declaration was not "
                 "materialized for this type");
          }
          reductionDeclSymbols.push_back(mlir::SymbolRefAttr::get(
              builder.getContext(), existingDecl.getSymName()));
          ++idx;
          continue;
        }
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
        // An intrinsic-operator USER reduction (declare reduction(+:t)) is
        // scoped by its owning scope, exactly like the defined-operator,
        // named, and named-shadowing paths, so two user declarations for the
        // same (operator, type) in different scopes get distinct ops instead of
        // colliding on the one global builtin name (a silent miscompile:
        // whichever lowers first wins the shared name and the other clause
        // binds the wrong combiner). This path is the lone unscoped one only
        // because the intrinsic-operator parse node carries no reduction
        // symbol; recover it by resolving from the current scope under the
        // operator's mangled name (MakeNameFromOperator, byte-identical to the
        // acceptance check in check-omp-structure). A resolved user reduction
        // is bound under its scoped name for ANY owner; an imported (mod-file)
        // one is materialized on demand first (a same-file one already exists
        // under the scoped name from the primary pass). Only when NO user
        // reduction is visible is this a genuine builtin, which keeps the
        // global name.
        if (semaCtx) {
          const semantics::DeclTypeSpec *opVarType =
              reductionSymbols[idx]->GetUltimate().GetType();
          parser::CharBlock mangledOpName =
              semantics::omp::MangledIntrinsicOperatorReductionName(
                  toParserIntrinsicOperator(intrinsicOp), *semaCtx);
          if (const semantics::Symbol *userSym =
                  semantics::omp::FindUserReductionSymbol(
                      converter.getCurrentScope(), mangledOpName, opVarType)) {
            const semantics::Symbol &ultimate = userSym->GetUltimate();
            std::string opName = ReductionProcessor::getScopedUserReductionName(
                converter, ultimate, namingType, isByRef);
            mlir::ModuleOp module = builder.getModule();
            auto existingDecl = module.lookupSymbol<OpType>(opName);
            // Separate compilation: materialize the imported reduction on
            // demand when its defining module is a mod file, then re-look it up
            // (a same-file op already exists here under the scoped name).
            if (!existingDecl && ultimate.owner().symbol() &&
                ultimate.owner().symbol()->test(
                    semantics::Symbol::Flag::ModFile)) {
              Fortran::lower::materializeUserReduction(
                  converter, *semaCtx, ultimate, opName, namingType, isByRef);
              existingDecl = module.lookupSymbol<OpType>(opName);
            }
            if (!existingDecl ||
                fir::unwrapRefType(existingDecl.getType()) !=
                    fir::unwrapRefType(namingType) ||
                isBoxedTrivialReduction) {
              TODO(
                  currentLocation,
                  "OpenMP user-defined intrinsic-operator reduction is not yet "
                  "lowered for this variable's shape (an unsupported element "
                  "type, a combiner-in-clause form, or an allocatable/pointer "
                  "trivial reduction; the last deferred to #186765)");
            }
            reductionDeclSymbols.push_back(mlir::SymbolRefAttr::get(
                builder.getContext(), existingDecl.getSymName()));
            ++idx;
            continue;
          }
        }
        // No user reduction is visible: a genuine builtin intrinsic reduction.
        // Reuse an existing op of the same global name if present; otherwise
        // fall through to createDeclareReduction below.
        if (auto existingDecl =
                builder.getModule().lookupSymbol<OpType>(reductionName)) {
          reductionDeclSymbols.push_back(mlir::SymbolRefAttr::get(
              builder.getContext(), existingDecl.getSymName()));
          ++idx;
          continue;
        }
      } else if (const auto *reductionIntrinsic =
                     std::get_if<omp::clause::ProcedureDesignator>(
                         &redOperator.u)) {
        if (!ReductionProcessor::supportedIntrinsicProcReduction(
                *reductionIntrinsic)) {
          // A user-defined named reduction (declare reduction(myred: ...)). The
          // clause references the (possibly USE-associated) reduction symbol;
          // bind its pre-materialized omp.declare_reduction op instead of
          // generating a new one.
          semantics::Symbol *sym = reductionIntrinsic->v.sym();
          if (!sym->GetUltimate()
                   .detailsIf<semantics::UserReductionDetails>()) {
            // Not a supported intrinsic proc (checked just above) and not a
            // user-defined reduction: a clean TODO, never a wrong binding. (A
            // named reduction reusing an intrinsic spelling such as `max` is
            // handled by the supportedIntrinsicProcReduction branch, not here.)
            //
            // For well-formed input this branch is unreachable:
            // CheckReductionOperator::visitDesignator (Semantics/
            // check-omp-structure.cpp) accepts a procedure-designator reduction
            // identifier only if its name is one of {max,min,iand,ior,ieor} or
            // it carries UserReductionDetails, and
            // supportedIntrinsicProcReduction above returns true for exactly
            // that same {max,min,iand,ior,ieor} set. The one gap: semantics
            // accepts by NAME while this path checks the INTRINSIC attribute,
            // so a user procedure that shadows an intrinsic reduction name
            // (with no declare reduction) can reach here. Keep a TODO rather
            // than emitFatalError to stay safe for that case.
            TODO(currentLocation, "Lowering unrecognised reduction type");
          }
          // Name the op from the resolved symbol's scoped ultimate (name,
          // owner) plus the per-type suffix, byte-identical to the
          // directive/materializer side, via getScopedUserReductionName.
          // Reductions of the same spelling in different scopes refer to
          // distinct ops, and a named reduction listing several types binds the
          // op for the variable's type instead of colliding on the first type's
          // op. namingType is the canonical element type (storage wrappers
          // stripped) so the by-ref suffix matches the directive.
          std::string reductionName =
              ReductionProcessor::getScopedUserReductionName(
                  converter, *sym, namingType, isByRef);
          mlir::ModuleOp module = builder.getModule();
          auto existingDecl = module.lookupSymbol<OpType>(reductionName);
          // Separate compilation: materialize an imported named reduction's op
          // on demand from the resolved symbol when its defining module is a
          // mod file, then re-look it up (same-file ops already exist here).
          const semantics::Symbol &namedUltimate = sym->GetUltimate();
          if (!existingDecl && semaCtx && namedUltimate.owner().symbol() &&
              namedUltimate.owner().symbol()->test(
                  semantics::Symbol::Flag::ModFile)) {
            Fortran::lower::materializeUserReduction(
                converter, *semaCtx, namedUltimate, reductionName, namingType,
                isByRef);
            existingDecl = module.lookupSymbol<OpType>(reductionName);
          }
          // The MLIR verifier does not type-check these ops, so this is the
          // only guard against binding a missing or mismatched declaration
          // (e.g. a named declare reduction that does not list the variable's
          // type). Compare unwrapped element types against namingType, the type
          // the op was named and created from. The guard fires when lowering
          // does not (yet) materialize an op for this variable's shape:
          //   - !existingDecl or a type mismatch: an imported reduction whose
          //     shape the materializer intentionally skips (a
          //     combiner-in-clause form, or an unsupported element type), so no
          //     op was created; better an error here than binding a wrong op.
          //   - isBoxedTrivialReduction: an allocatable/pointer reduction of a
          //     trivial element type, deferred to #186765.
          if (!existingDecl ||
              fir::unwrapRefType(existingDecl.getType()) !=
                  fir::unwrapRefType(namingType) ||
              isBoxedTrivialReduction) {
            TODO(currentLocation,
                 "OpenMP user-defined named reduction is not yet lowered for "
                 "this variable's shape (an unsupported element type, a "
                 "combiner-in-clause form, or an allocatable/pointer trivial "
                 "reduction; the last deferred to #186765)");
          }
          reductionDeclSymbols.push_back(mlir::SymbolRefAttr::get(
              builder.getContext(), existingDecl.getSymName()));
          ++idx;
          continue;
        }
        // A user-defined reduction may shadow a built-in intrinsic reduction
        // of the same name (max/min/iand/ior/ieor). Per the OpenMP spec, such
        // reduction has the same visibility as a variable declared at the same
        // location, so a visible declaration takes precedence over the
        // intrinsic. Semantics names it "op<name>" (MangleSpecialFunctions in
        // resolve-names). If one is visible in the current scope and supports
        // the variable's type, bind to the omp.declare_reduction op the
        // directive materialized for it instead of generating the intrinsic.
        semantics::Symbol *sym = reductionIntrinsic->v.sym();
        std::string mangledName = "op." + getRealName(sym).ToString();
        if (const semantics::Symbol *redSym =
                converter.getCurrentScope().FindSymbol(
                    parser::CharBlock{mangledName})) {
          const semantics::Symbol &ultimate = redSym->GetUltimate();
          const semantics::UserReductionDetails *userDetails =
              ultimate.detailsIf<semantics::UserReductionDetails>();
          const semantics::DeclTypeSpec *varType =
              reductionSymbols[idx]->GetUltimate().GetType();
          // A user-defined reduction shadows the intrinsic only for the types
          // it is declared for. If it does not cover this variable's type, the
          // user has not redefined the reduction for that type and the
          // implicit intrinsic reduction still applies, so fall through to it.
          if (userDetails && varType && userDetails->SupportsType(*varType)) {
            // The user declaration takes precedence over the intrinsic for this
            // type. A declaration listing several types (or several merged
            // declarations) is handled the same way as the operator and named
            // paths: the directive emits one op per type and the variable's
            // type selects the matching per-type name below. A USE-associated
            // shadowing reduction is found by FindSymbol as a use wrapper;
            // naming from its ultimate (name, owner) below binds the source
            // module's op, materialized on demand here for separate
            // compilation, exactly as the named path does. A renamed shadowing
            // intrinsic does not reach here: the renamed name resolves to the
            // intrinsic rather than the user reduction, so semantics rejects
            // the clause with a type-incompatibility error before lowering.
            std::string opName = ReductionProcessor::getScopedUserReductionName(
                converter, ultimate, namingType, isByRef);
            mlir::ModuleOp module = builder.getModule();
            auto existingDecl = module.lookupSymbol<OpType>(opName);
            // Separate compilation: materialize the imported shadowing
            // reduction on demand when its defining module is a mod file, then
            // re-look it up (same-file ops already exist here).
            if (!existingDecl && semaCtx && ultimate.owner().symbol() &&
                ultimate.owner().symbol()->test(
                    semantics::Symbol::Flag::ModFile)) {
              Fortran::lower::materializeUserReduction(
                  converter, *semaCtx, ultimate, opName, namingType, isByRef);
              existingDecl = module.lookupSymbol<OpType>(opName);
            }
            if (!existingDecl ||
                fir::unwrapRefType(existingDecl.getType()) !=
                    fir::unwrapRefType(namingType) ||
                isBoxedTrivialReduction) {
              TODO(currentLocation,
                   "OpenMP user-defined reduction declaration was not "
                   "materialized for this type");
            }
            reductionDeclSymbols.push_back(mlir::SymbolRefAttr::get(
                builder.getContext(), existingDecl.getSymName()));
            ++idx;
            continue;
          }
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
