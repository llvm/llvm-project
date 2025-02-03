//===-- PrivateReductionUtils.cpp -------------------------------*- C++ -*-===//
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

#include "PrivateReductionUtils.h"

#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Semantics/symbol.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Location.h"

static bool hasFinalization(const Fortran::semantics::Symbol &sym) {
  if (sym.has<Fortran::semantics::ObjectEntityDetails>())
    if (const Fortran::semantics::DeclTypeSpec *declTypeSpec = sym.GetType())
      if (const Fortran::semantics::DerivedTypeSpec *derivedTypeSpec =
              declTypeSpec->AsDerived())
        return Fortran::semantics::IsFinalizable(*derivedTypeSpec);
  return false;
}

static void createCleanupRegion(Fortran::lower::AbstractConverter &converter,
                                mlir::Location loc, mlir::Type argType,
                                mlir::Region &cleanupRegion,
                                const Fortran::semantics::Symbol *sym) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  assert(cleanupRegion.empty());
  mlir::Block *block = builder.createBlock(&cleanupRegion, cleanupRegion.end(),
                                           {argType}, {loc});
  builder.setInsertionPointToEnd(block);

  auto typeError = [loc]() {
    fir::emitFatalError(loc,
                        "Attempt to create an omp cleanup region "
                        "for a type that wasn't allocated",
                        /*genCrashDiag=*/true);
  };

  mlir::Type valTy = fir::unwrapRefType(argType);
  if (auto boxTy = mlir::dyn_cast_or_null<fir::BaseBoxType>(valTy)) {
    // TODO: what about undoing init of unboxed derived types?
    if (auto recTy = mlir::dyn_cast<fir::RecordType>(
            fir::unwrapSequenceType(fir::dyn_cast_ptrOrBoxEleTy(boxTy)))) {
      mlir::Type eleTy = boxTy.getEleTy();
      if (mlir::isa<fir::PointerType, fir::HeapType>(eleTy)) {
        mlir::Type mutableBoxTy =
            fir::ReferenceType::get(fir::BoxType::get(eleTy));
        mlir::Value converted =
            builder.createConvert(loc, mutableBoxTy, block->getArgument(0));
        if (recTy.getNumLenParams() > 0)
          TODO(loc, "Deallocate box with length parameters");
        fir::MutableBoxValue mutableBox{converted, /*lenParameters=*/{},
                                        /*mutableProperties=*/{}};
        Fortran::lower::genDeallocateIfAllocated(converter, mutableBox, loc);
        builder.create<mlir::omp::YieldOp>(loc);
        return;
      }
    }

    // TODO: just replace this whole body with
    // Fortran::lower::genDeallocateIfAllocated (not done now to avoid test
    // churn)

    mlir::Value arg = builder.loadIfRef(loc, block->getArgument(0));
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

  if (auto boxCharTy = mlir::dyn_cast<fir::BoxCharType>(argType)) {
    auto [addr, len] =
        fir::factory::CharacterExprHelper{builder, loc}.createUnboxChar(
            block->getArgument(0));

    // convert addr to a heap type so it can be used with fir::FreeMemOp
    auto refTy = mlir::cast<fir::ReferenceType>(addr.getType());
    auto heapTy = fir::HeapType::get(refTy.getEleTy());
    addr = builder.createConvert(loc, heapTy, addr);

    builder.create<fir::FreeMemOp>(loc, addr);
    builder.create<mlir::omp::YieldOp>(loc);
    return;
  }

  typeError();
}

fir::ShapeShiftOp Fortran::lower::omp::getShapeShift(fir::FirOpBuilder &builder,
                                                     mlir::Location loc,
                                                     mlir::Value box) {
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

// Initialize box newBox using moldBox. These should both have the same type and
// be boxes containing derived types e.g.
// fir.box<!fir.type<>>
// fir.box<!fir.heap<!fir.type<>>
// fir.box<!fir.heap<!fir.array<fir.type<>>>
// fir.class<...<!fir.type<>>>
// If the type doesn't match , this does nothing
static void initializeIfDerivedTypeBox(fir::FirOpBuilder &builder,
                                       mlir::Location loc, mlir::Value newBox,
                                       mlir::Value moldBox, bool hasInitializer,
                                       bool isFirstPrivate) {
  assert(moldBox.getType() == newBox.getType());
  fir::BoxType boxTy = mlir::dyn_cast<fir::BoxType>(newBox.getType());
  fir::ClassType classTy = mlir::dyn_cast<fir::ClassType>(newBox.getType());
  if (!boxTy && !classTy)
    return;

  // remove pointer and array types in the middle
  mlir::Type eleTy = boxTy ? boxTy.getElementType() : classTy.getEleTy();
  mlir::Type derivedTy = fir::unwrapRefType(eleTy);
  if (auto array = mlir::dyn_cast<fir::SequenceType>(derivedTy))
    derivedTy = array.getElementType();

  if (!fir::isa_derived(derivedTy))
    return;

  if (hasInitializer)
    fir::runtime::genDerivedTypeInitialize(builder, loc, newBox);

  if (hlfir::mayHaveAllocatableComponent(derivedTy) && !isFirstPrivate)
    fir::runtime::genDerivedTypeInitializeClone(builder, loc, newBox, moldBox);
}

static void getLengthParameters(fir::FirOpBuilder &builder, mlir::Location loc,
                                mlir::Value moldArg,
                                llvm::SmallVectorImpl<mlir::Value> &lenParams) {
  // We pass derived types unboxed and so are not self-contained entities.
  // Assume that unboxed derived types won't need length paramters.
  if (!hlfir::isFortranEntity(moldArg))
    return;

  hlfir::genLengthParameters(loc, builder, hlfir::Entity{moldArg}, lenParams);
  if (lenParams.empty())
    return;

  // The verifier for EmboxOp doesn't allow length parameters when the the
  // character already has static LEN. genLengthParameters may still return them
  // in this case.
  mlir::Type unwrappedType =
      fir::unwrapRefType(fir::unwrapSeqOrBoxedSeqType(moldArg.getType()));
  if (auto strTy = mlir::dyn_cast<fir::CharacterType>(unwrappedType)) {
    if (strTy.hasConstantLen())
      lenParams.resize(0);
  }
}

static bool
isDerivedTypeNeedingInitialization(const Fortran::semantics::Symbol &sym) {
  // Fortran::lower::hasDefaultInitialization returns false for ALLOCATABLE, so
  // re-implement here.
  // ignorePointer=true because either the pointer points to the same target as
  // the original variable, or it is uninitialized.
  if (const Fortran::semantics::DeclTypeSpec *declTypeSpec = sym.GetType())
    if (const Fortran::semantics::DerivedTypeSpec *derivedTypeSpec =
            declTypeSpec->AsDerived())
      return derivedTypeSpec->HasDefaultInitialization(
          /*ignoreAllocatable=*/false, /*ignorePointer=*/true);
  return false;
}

static mlir::Value generateZeroShapeForRank(fir::FirOpBuilder &builder,
                                            mlir::Location loc,
                                            mlir::Value moldArg) {
  mlir::Type moldType = fir::unwrapRefType(moldArg.getType());
  mlir::Type eleType = fir::dyn_cast_ptrOrBoxEleTy(moldType);
  fir::SequenceType seqTy =
      mlir::dyn_cast_if_present<fir::SequenceType>(eleType);
  if (!seqTy)
    return mlir::Value{};

  unsigned rank = seqTy.getShape().size();
  mlir::Value zero =
      builder.createIntegerConstant(loc, builder.getIndexType(), 0);
  mlir::SmallVector<mlir::Value> dims;
  dims.resize(rank, zero);
  mlir::Type shapeTy = fir::ShapeType::get(builder.getContext(), rank);
  return builder.create<fir::ShapeOp>(loc, shapeTy, dims);
}

void Fortran::lower::omp::populateByRefInitAndCleanupRegions(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Type argType, mlir::Value scalarInitValue, mlir::Block *initBlock,
    mlir::Value allocatedPrivVarArg, mlir::Value moldArg,
    mlir::Region &cleanupRegion, DeclOperationKind kind,
    const Fortran::semantics::Symbol *sym) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Type ty = fir::unwrapRefType(argType);
  builder.setInsertionPointToEnd(initBlock);
  auto yield = [&](mlir::Value ret) {
    builder.create<mlir::omp::YieldOp>(loc, ret);
  };

  if (isPrivatization(kind))
    assert(sym && "Symbol information is needed to privatize derived types");
  bool needsInitialization =
      sym ? isDerivedTypeNeedingInitialization(sym->GetUltimate()) : false;

  if (fir::isa_trivial(ty)) {
    builder.setInsertionPointToEnd(initBlock);

    if (scalarInitValue)
      builder.createStoreWithConvert(loc, scalarInitValue, allocatedPrivVarArg);
    yield(allocatedPrivVarArg);
    return;
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
  mlir::SmallVector<mlir::Value> lenParams;
  auto handleNullAllocatable = [&](mlir::Value boxAlloca,
                                   mlir::Value loadedMold) -> fir::IfOp {
    mlir::Value addr = builder.create<fir::BoxAddrOp>(loc, loadedMold);
    mlir::Value isNotAllocated = builder.genIsNullAddr(loc, addr);
    fir::IfOp ifOp = builder.create<fir::IfOp>(loc, isNotAllocated,
                                               /*withElseRegion=*/true);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    // Just embox the null address and return.
    // We have to give the embox a shape so that the LLVM box structure has the
    // right rank. This returns an empty value if the types don't match.
    mlir::Value shape = generateZeroShapeForRank(builder, loc, moldArg);

    mlir::Value nullBox =
        builder.create<fir::EmboxOp>(loc, ty, addr, shape,
                                     /*slice=*/mlir::Value{}, lenParams);
    builder.create<fir::StoreOp>(loc, nullBox, boxAlloca);
    return ifOp;
  };

  // all arrays are boxed
  if (auto boxTy = mlir::dyn_cast_or_null<fir::BaseBoxType>(ty)) {
    bool isAllocatableOrPointer =
        mlir::isa<fir::HeapType, fir::PointerType>(boxTy.getEleTy());

    builder.setInsertionPointToEnd(initBlock);
    mlir::Value boxAlloca = allocatedPrivVarArg;

    moldArg = builder.loadIfRef(loc, moldArg);
    getLengthParameters(builder, loc, moldArg, lenParams);

    // The initial state of a private pointer is undefined so we don't need to
    // match the mold argument (OpenMP 5.2 end of page 106).
    if (isPrivatization(kind) &&
        mlir::isa<fir::PointerType>(boxTy.getEleTy())) {
      // we need a shape with the right rank so that the embox op is lowered
      // to an llvm struct of the right type. This returns nullptr if the types
      // aren't right.
      mlir::Value shape = generateZeroShapeForRank(builder, loc, moldArg);
      // Just incase, do initialize the box with a null value
      mlir::Value null = builder.createNullConstant(loc, boxTy.getEleTy());
      mlir::Value nullBox;
      nullBox = builder.create<fir::EmboxOp>(
          loc, boxTy, null, shape, /*slice=*/mlir::Value{}, lenParams);
      builder.create<fir::StoreOp>(loc, nullBox, boxAlloca);
      yield(boxAlloca);
      return;
    }

    mlir::Type innerTy = fir::unwrapRefType(boxTy.getEleTy());
    bool isDerived = fir::isa_derived(innerTy);
    bool isChar = fir::isa_char(innerTy);
    if (fir::isa_trivial(innerTy) || isDerived || isChar) {
      // boxed non-sequence value e.g. !fir.box<!fir.heap<i32>>
      if (!isAllocatableOrPointer && !isDerived)
        TODO(loc, "Reduction/Privatization of non-allocatable trivial or "
                  "character typed box");

      if ((isDerived || isChar) && (isReduction(kind) || scalarInitValue))
        TODO(loc, "Reduction of an unsupported boxed type");

      fir::IfOp ifUnallocated{nullptr};
      if (isAllocatableOrPointer) {
        ifUnallocated = handleNullAllocatable(boxAlloca, moldArg);
        builder.setInsertionPointToStart(
            &ifUnallocated.getElseRegion().front());
      }

      mlir::Value valAlloc = builder.createHeapTemporary(
          loc, innerTy, /*name=*/{}, /*shape=*/{}, lenParams);
      if (scalarInitValue)
        builder.createStoreWithConvert(loc, scalarInitValue, valAlloc);
      mlir::Value box = builder.create<fir::EmboxOp>(
          loc, ty, valAlloc, /*shape=*/mlir::Value{}, /*slice=*/mlir::Value{},
          lenParams);
      initializeIfDerivedTypeBox(
          builder, loc, box, moldArg, needsInitialization,
          /*isFirstPrivate=*/kind == DeclOperationKind::FirstPrivate);
      fir::StoreOp lastOp = builder.create<fir::StoreOp>(loc, box, boxAlloca);

      createCleanupRegion(converter, loc, argType, cleanupRegion, sym);

      if (ifUnallocated)
        builder.setInsertionPointAfter(ifUnallocated);
      else
        builder.setInsertionPointAfter(lastOp);
      yield(boxAlloca);
      return;
    }

    innerTy = fir::extractSequenceType(boxTy);
    if (!innerTy || !mlir::isa<fir::SequenceType>(innerTy))
      TODO(loc, "Unsupported boxed type for reduction/privatization");

    moldArg = builder.loadIfRef(loc, moldArg);
    getLengthParameters(builder, loc, moldArg, lenParams);

    fir::IfOp ifUnallocated{nullptr};
    if (isAllocatableOrPointer) {
      ifUnallocated = handleNullAllocatable(boxAlloca, moldArg);
      builder.setInsertionPointToStart(&ifUnallocated.getElseRegion().front());
    }

    // Create the private copy from the initial fir.box:
    mlir::Value loadedBox = builder.loadIfRef(loc, moldArg);
    hlfir::Entity source = hlfir::Entity{loadedBox};

    // Special case for (possibly allocatable) arrays of polymorphic types
    // e.g. !fir.class<!fir.heap<!fir.array<?x!fir.type<>>>>
    if (source.isPolymorphic()) {
      fir::ShapeShiftOp shape = getShapeShift(builder, loc, source);
      mlir::Type arrayType = source.getElementOrSequenceType();
      mlir::Value allocatedArray = builder.create<fir::AllocMemOp>(
          loc, arrayType, /*typeparams=*/mlir::ValueRange{},
          shape.getExtents());
      mlir::Value firClass = builder.create<fir::EmboxOp>(
          loc, source.getType(), allocatedArray, shape);
      initializeIfDerivedTypeBox(
          builder, loc, firClass, source, needsInitialization,
          /*isFirstprivate=*/kind == DeclOperationKind::FirstPrivate);
      builder.create<fir::StoreOp>(loc, firClass, allocatedPrivVarArg);
      if (ifUnallocated)
        builder.setInsertionPointAfter(ifUnallocated);
      yield(allocatedPrivVarArg);
      mlir::OpBuilder::InsertionGuard guard(builder);
      createCleanupRegion(converter, loc, argType, cleanupRegion, sym);
      return;
    }

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
      createCleanupRegion(converter, loc, argType, cleanupRegion, sym);
    } else {
      assert(!isAllocatableOrPointer &&
             "Pointer-like arrays must be heap allocated");
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

    if (scalarInitValue)
      builder.create<hlfir::AssignOp>(loc, scalarInitValue, box);

    initializeIfDerivedTypeBox(builder, loc, box, moldArg, needsInitialization,
                               /*isFirstPrivate=*/kind ==
                                   DeclOperationKind::FirstPrivate);

    builder.create<fir::StoreOp>(loc, box, boxAlloca);
    if (ifUnallocated)
      builder.setInsertionPointAfter(ifUnallocated);
    yield(boxAlloca);
    return;
  }

  if (auto boxCharTy = mlir::dyn_cast<fir::BoxCharType>(argType)) {
    mlir::Type eleTy = boxCharTy.getEleTy();
    builder.setInsertionPointToStart(initBlock);
    fir::factory::CharacterExprHelper charExprHelper{builder, loc};
    auto [addr, len] = charExprHelper.createUnboxChar(moldArg);

    // Using heap temporary so that
    // 1) It is safe to use privatization inside of big loops.
    // 2) The lifetime can outlive the current stack frame for delayed task
    // execution.
    // We can't always allocate a boxchar implicitly as the type of the
    // omp.private because the allocation potentially needs the length
    // parameters fetched above.
    // TODO: this deviates from the intended design for delayed task execution.
    mlir::Value privateAddr = builder.createHeapTemporary(
        loc, eleTy, /*name=*/{}, /*shape=*/{}, /*lenParams=*/len);
    mlir::Value boxChar = charExprHelper.createEmboxChar(privateAddr, len);

    createCleanupRegion(converter, loc, argType, cleanupRegion, sym);

    builder.setInsertionPointToEnd(initBlock);
    yield(boxChar);
    return;
  }

  if (fir::isa_derived(ty)) {
    builder.setInsertionPointToStart(initBlock);
    mlir::Type boxedTy = fir::BoxType::get(ty);
    mlir::Value newBox =
        builder.create<fir::EmboxOp>(loc, boxedTy, allocatedPrivVarArg);
    mlir::Value moldBox = builder.create<fir::EmboxOp>(loc, boxedTy, moldArg);
    initializeIfDerivedTypeBox(
        builder, loc, newBox, moldBox, needsInitialization,
        /*isFirstPrivate=*/kind == DeclOperationKind::FirstPrivate);

    if (sym && hasFinalization(*sym))
      createCleanupRegion(converter, loc, argType, cleanupRegion, sym);

    builder.setInsertionPointToEnd(initBlock);
    yield(allocatedPrivVarArg);
    return;
  }

  TODO(loc,
       "creating reduction/privatization init region for unsupported type");
  return;
}
