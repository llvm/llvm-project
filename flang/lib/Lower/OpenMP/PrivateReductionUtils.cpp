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

#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Location.h"

static void createCleanupRegion(fir::FirOpBuilder &builder, mlir::Location loc,
                                mlir::Type argType,
                                mlir::Region &cleanupRegion) {
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
    if (!mlir::isa<fir::HeapType, fir::PointerType>(boxTy.getEleTy())) {
      mlir::Type innerTy = fir::extractSequenceType(boxTy);
      if (!mlir::isa<fir::SequenceType>(innerTy))
        typeError();
    }

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

void Fortran::lower::omp::populateByRefInitAndCleanupRegions(
    fir::FirOpBuilder &builder, mlir::Location loc, mlir::Type argType,
    mlir::Value scalarInitValue, mlir::Block *initBlock,
    mlir::Value allocatedPrivVarArg, mlir::Value moldArg,
    mlir::Region &cleanupRegion, bool isPrivate) {
  mlir::Type ty = fir::unwrapRefType(argType);
  builder.setInsertionPointToEnd(initBlock);
  auto yield = [&](mlir::Value ret) {
    builder.create<mlir::omp::YieldOp>(loc, ret);
  };

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
    // just embox the null address and return
    mlir::Value nullBox =
        builder.create<fir::EmboxOp>(loc, ty, addr, /*shape=*/mlir::Value{},
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

    // The initial state of a private pointer is undefined so we don't need to
    // match the mold argument (OpenMP 5.2 end of page 106).
    if (isPrivate && mlir::isa<fir::PointerType>(boxTy.getEleTy())) {
      // Just incase, do initialize the box with a null value
      mlir::Value null = builder.createNullConstant(loc, boxTy.getEleTy());
      mlir::Value nullBox = builder.create<fir::EmboxOp>(loc, boxTy, null);
      builder.create<fir::StoreOp>(loc, nullBox, boxAlloca);
      yield(boxAlloca);
      return;
    }

    moldArg = builder.loadIfRef(loc, moldArg);
    hlfir::genLengthParameters(loc, builder, hlfir::Entity{moldArg}, lenParams);

    mlir::Type innerTy = fir::unwrapRefType(boxTy.getEleTy());
    bool isChar = fir::isa_char(innerTy);
    if (fir::isa_trivial(innerTy) || isChar) {
      // boxed non-sequence value e.g. !fir.box<!fir.heap<i32>>
      if (!isAllocatableOrPointer)
        TODO(loc,
             "Reduction/Privatization of non-allocatable trivial typed box");

      fir::IfOp ifUnallocated = handleNullAllocatable(boxAlloca, moldArg);

      builder.setInsertionPointToStart(&ifUnallocated.getElseRegion().front());
      mlir::Value valAlloc = builder.createHeapTemporary(
          loc, innerTy, /*name=*/{}, /*shape=*/{}, lenParams);
      if (scalarInitValue)
        builder.createStoreWithConvert(loc, scalarInitValue, valAlloc);
      mlir::Value box = builder.create<fir::EmboxOp>(
          loc, ty, valAlloc, /*shape=*/mlir::Value{}, /*slice=*/mlir::Value{},
          lenParams);
      builder.create<fir::StoreOp>(loc, box, boxAlloca);

      createCleanupRegion(builder, loc, argType, cleanupRegion);
      builder.setInsertionPointAfter(ifUnallocated);
      yield(boxAlloca);
      return;
    }
    innerTy = fir::extractSequenceType(boxTy);
    if (!innerTy || !mlir::isa<fir::SequenceType>(innerTy))
      TODO(loc, "Unsupported boxed type for reduction/privatization");

    moldArg = builder.loadIfRef(loc, moldArg);
    hlfir::genLengthParameters(loc, builder, hlfir::Entity{moldArg}, lenParams);

    fir::IfOp ifUnallocated{nullptr};
    if (isAllocatableOrPointer) {
      ifUnallocated = handleNullAllocatable(boxAlloca, moldArg);
      builder.setInsertionPointToStart(&ifUnallocated.getElseRegion().front());
    }

    // Create the private copy from the initial fir.box:
    mlir::Value loadedBox = builder.loadIfRef(loc, moldArg);
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
      createCleanupRegion(builder, loc, argType, cleanupRegion);
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

    createCleanupRegion(builder, loc, argType, cleanupRegion);

    builder.setInsertionPointToEnd(initBlock);
    yield(boxChar);
    return;
  }

  TODO(loc,
       "creating reduction/privatization init region for unsupported type");
  return;
}
