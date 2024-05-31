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
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/FatalError.h"
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
  semantics::Symbol *sym = pd.v.id();
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
    if (auto cplxTy = mlir::dyn_cast<fir::ComplexType>(type)) {
      mlir::Type realTy =
          lower::convertReal(builder.getContext(), cplxTy.getFKind());
      mlir::Value initRe = builder.createRealConstant(
          loc, realTy, getOperationIdentity(redId, loc));
      mlir::Value initIm = builder.createRealConstant(loc, realTy, 0);

      return fir::factory::Complex{builder, loc}.createComplex(type, initRe,
                                                               initIm);
    }
    if (mlir::isa<mlir::FloatType>(type))
      return builder.create<mlir::arith::ConstantOp>(
          loc, type,
          builder.getFloatAttr(type, (double)getOperationIdentity(redId, loc)));

    if (mlir::isa<fir::LogicalType>(type)) {
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

/// Generate a fir::ShapeShift op describing the provided boxed array.
static fir::ShapeShiftOp getShapeShift(fir::FirOpBuilder &builder,
                                       mlir::Location loc, mlir::Value box) {
  fir::SequenceType sequenceType = mlir::cast<fir::SequenceType>(
      hlfir::getFortranElementOrSequenceType(box.getType()));
  const unsigned rank = sequenceType.getDimension();
  llvm::SmallVector<mlir::Value> lbAndExtents;
  lbAndExtents.reserve(rank * 2);

  mlir::Type idxTy = builder.getIndexType();
  for (unsigned i = 0; i < rank; ++i) {
    // TODO: ideally we want to hoist box reads out of the critical section.
    // We could do this by having box dimensions in block arguments like
    // OpenACC does
    mlir::Value dim = builder.createIntegerConstant(loc, idxTy, i);
    auto dimInfo =
        builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, box, dim);
    lbAndExtents.push_back(dimInfo.getLowerBound());
    lbAndExtents.push_back(dimInfo.getExtent());
  }

  auto shapeShiftTy = fir::ShapeShiftType::get(builder.getContext(), rank);
  auto shapeShift =
      builder.create<fir::ShapeShiftOp>(loc, shapeShiftTy, lbAndExtents);
  return shapeShift;
}

/// Create reduction combiner region for reduction variables which are boxed
/// arrays
static void genBoxCombiner(fir::FirOpBuilder &builder, mlir::Location loc,
                           ReductionProcessor::ReductionIdentifier redId,
                           fir::BaseBoxType boxTy, mlir::Value lhs,
                           mlir::Value rhs) {
  fir::SequenceType seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(
      fir::unwrapRefType(boxTy.getEleTy()));
  fir::HeapType heapTy =
      mlir::dyn_cast_or_null<fir::HeapType>(boxTy.getEleTy());
  if ((!seqTy || seqTy.hasUnknownShape()) && !heapTy)
    TODO(loc, "Unsupported boxed type in OpenMP reduction");

  // load fir.ref<fir.box<...>>
  mlir::Value lhsAddr = lhs;
  lhs = builder.create<fir::LoadOp>(loc, lhs);
  rhs = builder.create<fir::LoadOp>(loc, rhs);

  if (heapTy && !seqTy) {
    // get box contents (heap pointers)
    lhs = builder.create<fir::BoxAddrOp>(loc, lhs);
    rhs = builder.create<fir::BoxAddrOp>(loc, rhs);
    mlir::Value lhsValAddr = lhs;

    // load heap pointers
    lhs = builder.create<fir::LoadOp>(loc, lhs);
    rhs = builder.create<fir::LoadOp>(loc, rhs);

    mlir::Value result = ReductionProcessor::createScalarCombiner(
        builder, loc, redId, heapTy.getEleTy(), lhs, rhs);
    builder.create<fir::StoreOp>(loc, result, lhsValAddr);
    builder.create<mlir::omp::YieldOp>(loc, lhsAddr);
    return;
  }

  fir::ShapeShiftOp shapeShift = getShapeShift(builder, loc, lhs);

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

static void
createReductionCleanupRegion(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::omp::DeclareReductionOp &reductionDecl) {
  mlir::Type redTy = reductionDecl.getType();

  mlir::Region &cleanupRegion = reductionDecl.getCleanupRegion();
  assert(cleanupRegion.empty());
  mlir::Block *block =
      builder.createBlock(&cleanupRegion, cleanupRegion.end(), {redTy}, {loc});
  builder.setInsertionPointToEnd(block);

  auto typeError = [loc]() {
    fir::emitFatalError(loc,
                        "Attempt to create an omp reduction cleanup region "
                        "for a type that wasn't allocated",
                        /*genCrashDiag=*/true);
  };

  mlir::Type valTy = fir::unwrapRefType(redTy);
  if (auto boxTy = mlir::dyn_cast_or_null<fir::BaseBoxType>(valTy)) {
    if (!mlir::isa<fir::HeapType>(boxTy.getEleTy())) {
      mlir::Type innerTy = fir::extractSequenceType(boxTy);
      if (!mlir::isa<fir::SequenceType>(innerTy))
        typeError();
    }

    mlir::Value arg = block->getArgument(0);
    arg = builder.loadIfRef(loc, arg);
    assert(mlir::isa<fir::BaseBoxType>(arg.getType()));

    // Deallocate box
    // The FIR type system doesn't nesecarrily know that this is a mutable box
    // if we allocated the thread local array on the heap to avoid looped stack
    // allocations.
    mlir::Value addr =
        hlfir::genVariableRawAddress(loc, builder, hlfir::Entity{arg});
    mlir::Value isAllocated = builder.genIsNotNullAddr(loc, addr);
    fir::IfOp ifOp =
        builder.create<fir::IfOp>(loc, isAllocated, /*withElseRegion=*/false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    mlir::Value cast = builder.createConvert(
        loc, fir::HeapType::get(fir::dyn_cast_ptrEleTy(addr.getType())), addr);
    builder.create<fir::FreeMemOp>(loc, cast);

    builder.setInsertionPointAfter(ifOp);
    builder.create<mlir::omp::YieldOp>(loc);
    return;
  }

  typeError();
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

static mlir::Value
createReductionInitRegion(fir::FirOpBuilder &builder, mlir::Location loc,
                          mlir::omp::DeclareReductionOp &reductionDecl,
                          const ReductionProcessor::ReductionIdentifier redId,
                          mlir::Type type, bool isByRef) {
  mlir::Type ty = fir::unwrapRefType(type);
  mlir::Value initValue = ReductionProcessor::getReductionInitValue(
      loc, unwrapSeqOrBoxedType(ty), redId, builder);

  if (fir::isa_trivial(ty)) {
    if (isByRef) {
      mlir::Value alloca = builder.create<fir::AllocaOp>(loc, ty);
      builder.createStoreWithConvert(loc, initValue, alloca);
      return alloca;
    }
    // by val
    return initValue;
  }

  // check if an allocatable box is unallocated. If so, initialize the boxAlloca
  // to be unallocated e.g.
  // %box_alloca = fir.alloca !fir.box<!fir.heap<...>>
  // %addr = fir.box_addr %box
  // if (%addr == 0) {
  //   %nullbox = fir.embox %addr
  //   fir.store %nullbox to %box_alloca
  // } else {
  //   // ...
  //   fir.store %something to %box_alloca
  // }
  // omp.yield %box_alloca
  mlir::Value blockArg =
      builder.loadIfRef(loc, builder.getBlock()->getArgument(0));
  auto handleNullAllocatable = [&](mlir::Value boxAlloca) -> fir::IfOp {
    mlir::Value addr = builder.create<fir::BoxAddrOp>(loc, blockArg);
    mlir::Value isNotAllocated = builder.genIsNullAddr(loc, addr);
    fir::IfOp ifOp = builder.create<fir::IfOp>(loc, isNotAllocated,
                                               /*withElseRegion=*/true);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    // just embox the null address and return
    mlir::Value nullBox = builder.create<fir::EmboxOp>(loc, ty, addr);
    builder.create<fir::StoreOp>(loc, nullBox, boxAlloca);
    return ifOp;
  };

  // all arrays are boxed
  if (auto boxTy = mlir::dyn_cast_or_null<fir::BaseBoxType>(ty)) {
    assert(isByRef && "passing boxes by value is unsupported");
    bool isAllocatable = mlir::isa<fir::HeapType>(boxTy.getEleTy());
    mlir::Value boxAlloca = builder.create<fir::AllocaOp>(loc, ty);
    mlir::Type innerTy = fir::unwrapRefType(boxTy.getEleTy());
    if (fir::isa_trivial(innerTy)) {
      // boxed non-sequence value e.g. !fir.box<!fir.heap<i32>>
      if (!isAllocatable)
        TODO(loc, "Reduction of non-allocatable trivial typed box");

      fir::IfOp ifUnallocated = handleNullAllocatable(boxAlloca);

      builder.setInsertionPointToStart(&ifUnallocated.getElseRegion().front());
      mlir::Value valAlloc = builder.create<fir::AllocMemOp>(loc, innerTy);
      builder.createStoreWithConvert(loc, initValue, valAlloc);
      mlir::Value box = builder.create<fir::EmboxOp>(loc, ty, valAlloc);
      builder.create<fir::StoreOp>(loc, box, boxAlloca);

      auto insPt = builder.saveInsertionPoint();
      createReductionCleanupRegion(builder, loc, reductionDecl);
      builder.restoreInsertionPoint(insPt);
      builder.setInsertionPointAfter(ifUnallocated);
      return boxAlloca;
    }
    innerTy = fir::extractSequenceType(boxTy);
    if (!mlir::isa<fir::SequenceType>(innerTy))
      TODO(loc, "Unsupported boxed type for reduction");

    fir::IfOp ifUnallocated{nullptr};
    if (isAllocatable) {
      ifUnallocated = handleNullAllocatable(boxAlloca);
      builder.setInsertionPointToStart(&ifUnallocated.getElseRegion().front());
    }

    // Create the private copy from the initial fir.box:
    mlir::Value loadedBox = builder.loadIfRef(loc, blockArg);
    hlfir::Entity source = hlfir::Entity{loadedBox};

    // Allocating on the heap in case the whole reduction is nested inside of a
    // loop
    // TODO: compare performance here to using allocas - this could be made to
    // work by inserting stacksave/stackrestore around the reduction in
    // openmpirbuilder
    auto [temp, needsDealloc] = createTempFromMold(loc, builder, source);
    // if needsDealloc isn't statically false, add cleanup region. Always
    // do this for allocatable boxes because they might have been re-allocated
    // in the body of the loop/parallel region

    std::optional<int64_t> cstNeedsDealloc =
        fir::getIntIfConstant(needsDealloc);
    assert(cstNeedsDealloc.has_value() &&
           "createTempFromMold decides this statically");
    if (cstNeedsDealloc.has_value() && *cstNeedsDealloc != false) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      createReductionCleanupRegion(builder, loc, reductionDecl);
    } else {
      assert(!isAllocatable && "Allocatable arrays must be heap allocated");
    }

    // Put the temporary inside of a box:
    // hlfir::genVariableBox doesn't handle non-default lower bounds
    mlir::Value box;
    fir::ShapeShiftOp shapeShift = getShapeShift(builder, loc, loadedBox);
    mlir::Type boxType = loadedBox.getType();
    if (mlir::isa<fir::BaseBoxType>(temp.getType()))
      // the box created by the declare form createTempFromMold is missing lower
      // bounds info
      box = builder.create<fir::ReboxOp>(loc, boxType, temp, shapeShift,
                                         /*shift=*/mlir::Value{});
    else
      box = builder.create<fir::EmboxOp>(
          loc, boxType, temp, shapeShift,
          /*slice=*/mlir::Value{},
          /*typeParams=*/llvm::ArrayRef<mlir::Value>{});

    builder.create<hlfir::AssignOp>(loc, initValue, box);
    builder.create<fir::StoreOp>(loc, box, boxAlloca);
    if (ifUnallocated)
      builder.setInsertionPointAfter(ifUnallocated);
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
      createReductionInitRegion(builder, loc, decl, redId, type, isByRef);
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

void ReductionProcessor::addDeclareReduction(
    mlir::Location currentLocation, lower::AbstractConverter &converter,
    const omp::clause::Reduction &reduction,
    llvm::SmallVectorImpl<mlir::Value> &reductionVars,
    llvm::SmallVectorImpl<bool> &reduceVarByRef,
    llvm::SmallVectorImpl<mlir::Attribute> &reductionDeclSymbols,
    llvm::SmallVectorImpl<const semantics::Symbol *> *reductionSymbols) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  if (std::get<std::optional<omp::clause::Reduction::ReductionModifier>>(
          reduction.t))
    TODO(currentLocation, "Reduction modifiers are not supported");

  mlir::omp::DeclareReductionOp decl;
  const auto &redOperatorList{
      std::get<omp::clause::Reduction::ReductionIdentifiers>(reduction.t)};
  assert(redOperatorList.size() == 1 && "Expecting single operator");
  const auto &redOperator = redOperatorList.front();
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
    const semantics::Symbol *symbol = object.id();
    if (reductionSymbols)
      reductionSymbols->push_back(symbol);
    mlir::Value symVal = converter.getSymbolAddress(*symbol);
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
          builder.create<fir::AllocaOp>(currentLocation, box.getType());
      builder.create<fir::StoreOp>(currentLocation, box, alloca);

      symVal = alloca;
    } else if (mlir::isa<fir::BaseBoxType>(symVal.getType())) {
      // boxed arrays are passed as values not by reference. Unfortunately,
      // we can't pass a box by value to omp.redution_declare, so turn it
      // into a reference

      auto alloca =
          builder.create<fir::AllocaOp>(currentLocation, symVal.getType());
      builder.create<fir::StoreOp>(currentLocation, symVal, alloca);
      symVal = alloca;
    } else if (auto declOp = symVal.getDefiningOp<hlfir::DeclareOp>()) {
      symVal = declOp.getBase();
    }

    // this isn't the same as the by-val and by-ref passing later in the
    // pipeline. Both styles assume that the variable is a reference at
    // this point
    assert(mlir::isa<fir::ReferenceType>(symVal.getType()) &&
           "reduction input var is a reference");

    reductionVars.push_back(symVal);
    reduceVarByRef.push_back(doReductionByRef(symVal));
  }

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

    for (auto [symVal, isByRef] : llvm::zip(reductionVars, reduceVarByRef)) {
      auto redType = mlir::cast<fir::ReferenceType>(symVal.getType());
      const auto &kindMap = firOpBuilder.getKindMap();
      if (mlir::isa<fir::LogicalType>(redType.getEleTy()))
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
      for (auto [symVal, isByRef] : llvm::zip(reductionVars, reduceVarByRef)) {
        auto redType = mlir::cast<fir::ReferenceType>(symVal.getType());
        if (!redType.getEleTy().isIntOrIndexOrFloat())
          TODO(currentLocation,
               "Reduction of some types is not supported for intrinsics");
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

const semantics::SourceName
ReductionProcessor::getRealName(const semantics::Symbol *symbol) {
  return symbol->GetUltimate().name();
}

const semantics::SourceName
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
