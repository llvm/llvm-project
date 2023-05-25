//===-- Optimizer/Builder/TemporaryStorage.cpp ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implementation of utility data structures to create and manipulate temporary
// storages to stack Fortran values or pointers in HLFIR.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/TemporaryStorage.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"

//===----------------------------------------------------------------------===//
// fir::factory::Counter implementation.
//===----------------------------------------------------------------------===//

fir::factory::Counter::Counter(mlir::Location loc, fir::FirOpBuilder &builder,
                               mlir::Value initialValue,
                               bool canCountThroughLoops)
    : canCountThroughLoops{canCountThroughLoops}, initialValue{initialValue} {
  mlir::Type type = initialValue.getType();
  one = builder.createIntegerConstant(loc, type, 1);
  if (canCountThroughLoops) {
    index = builder.createTemporary(loc, type);
    builder.create<fir::StoreOp>(loc, initialValue, index);
  } else {
    index = initialValue;
  }
}

mlir::Value
fir::factory::Counter::getAndIncrementIndex(mlir::Location loc,
                                            fir::FirOpBuilder &builder) {
  if (canCountThroughLoops) {
    mlir::Value indexValue = builder.create<fir::LoadOp>(loc, index);
    mlir::Value newValue =
        builder.create<mlir::arith::AddIOp>(loc, indexValue, one);
    builder.create<fir::StoreOp>(loc, newValue, index);
    return indexValue;
  }
  mlir::Value indexValue = index;
  index = builder.create<mlir::arith::AddIOp>(loc, indexValue, one);
  return indexValue;
}

void fir::factory::Counter::reset(mlir::Location loc,
                                  fir::FirOpBuilder &builder) {
  if (canCountThroughLoops)
    builder.create<fir::StoreOp>(loc, initialValue, index);
  else
    index = initialValue;
}

//===----------------------------------------------------------------------===//
// fir::factory::HomogeneousScalarStack implementation.
//===----------------------------------------------------------------------===//

fir::factory::HomogeneousScalarStack::HomogeneousScalarStack(
    mlir::Location loc, fir::FirOpBuilder &builder,
    fir::SequenceType declaredType, mlir::Value extent,
    llvm::ArrayRef<mlir::Value> lengths, bool allocateOnHeap,
    bool stackThroughLoops, llvm::StringRef tempName)
    : allocateOnHeap{allocateOnHeap},
      counter{loc, builder,
              builder.createIntegerConstant(loc, builder.getIndexType(), 1),
              stackThroughLoops} {
  // Allocate the temporary storage.
  llvm::SmallVector<mlir::Value, 1> extents{extent};
  mlir::Value tempStorage;
  if (allocateOnHeap)
    tempStorage = builder.createHeapTemporary(loc, declaredType, tempName,
                                              extents, lengths);
  else
    tempStorage =
        builder.createTemporary(loc, declaredType, tempName, extents, lengths);

  mlir::Value shape = builder.genShape(loc, extents);
  temp = builder
             .create<hlfir::DeclareOp>(loc, tempStorage, tempName, shape,
                                       lengths, fir::FortranVariableFlagsAttr{})
             .getBase();
}

void fir::factory::HomogeneousScalarStack::pushValue(mlir::Location loc,
                                                     fir::FirOpBuilder &builder,
                                                     mlir::Value value) {
  hlfir::Entity entity{value};
  assert(entity.isScalar() && "cannot use inlined temp with array");
  mlir::Value indexValue = counter.getAndIncrementIndex(loc, builder);
  hlfir::Entity tempElement = hlfir::getElementAt(
      loc, builder, hlfir::Entity{temp}, mlir::ValueRange{indexValue});
  // TODO: "copy" would probably be better than assign to ensure there are no
  // side effects (user assignments, temp, lhs finalization)?
  // This only makes a difference for derived types, and for now derived types
  // will use the runtime strategy to avoid any bad behaviors. So the todo
  // below should not get hit but is added as a remainder/safety.
  if (!entity.hasIntrinsicType())
    TODO(loc, "creating inlined temporary stack for derived types");
  builder.create<hlfir::AssignOp>(loc, value, tempElement);
}

void fir::factory::HomogeneousScalarStack::resetFetchPosition(
    mlir::Location loc, fir::FirOpBuilder &builder) {
  counter.reset(loc, builder);
}

mlir::Value
fir::factory::HomogeneousScalarStack::fetch(mlir::Location loc,
                                            fir::FirOpBuilder &builder) {
  mlir::Value indexValue = counter.getAndIncrementIndex(loc, builder);
  hlfir::Entity tempElement = hlfir::getElementAt(
      loc, builder, hlfir::Entity{temp}, mlir::ValueRange{indexValue});
  return hlfir::loadTrivialScalar(loc, builder, tempElement);
}

void fir::factory::HomogeneousScalarStack::destroy(mlir::Location loc,
                                                   fir::FirOpBuilder &builder) {
  if (allocateOnHeap) {
    auto declare = temp.getDefiningOp<hlfir::DeclareOp>();
    assert(declare && "temp must have been declared");
    builder.create<fir::FreeMemOp>(loc, declare.getMemref());
  }
}

hlfir::Entity fir::factory::HomogeneousScalarStack::moveStackAsArrayExpr(
    mlir::Location loc, fir::FirOpBuilder &builder) {
  mlir::Value mustFree = builder.createBool(loc, allocateOnHeap);
  auto hlfirExpr = builder.create<hlfir::AsExprOp>(loc, temp, mustFree);
  return hlfir::Entity{hlfirExpr};
}

//===----------------------------------------------------------------------===//
// fir::factory::SimpleCopy implementation.
//===----------------------------------------------------------------------===//

fir::factory::SimpleCopy::SimpleCopy(mlir::Location loc,
                                     fir::FirOpBuilder &builder,
                                     hlfir::Entity source,
                                     llvm::StringRef tempName) {
  // Use hlfir.as_expr and hlfir.associate to create a copy and leave
  // bufferization deals with how best to make the copy.
  if (source.isVariable())
    source = hlfir::Entity{builder.create<hlfir::AsExprOp>(loc, source)};
  copy = hlfir::genAssociateExpr(loc, builder, source,
                                 source.getFortranElementType(), tempName);
}

void fir::factory::SimpleCopy::destroy(mlir::Location loc,
                                       fir::FirOpBuilder &builder) {
  builder.create<hlfir::EndAssociateOp>(loc, copy);
}
