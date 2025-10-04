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
#include "flang/Optimizer/Builder/Runtime/TemporaryStack.h"
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
    fir::StoreOp::create(builder, loc, initialValue, index);
  } else {
    index = initialValue;
  }
}

mlir::Value
fir::factory::Counter::getAndIncrementIndex(mlir::Location loc,
                                            fir::FirOpBuilder &builder) {
  if (canCountThroughLoops) {
    mlir::Value indexValue = fir::LoadOp::create(builder, loc, index);
    mlir::Value newValue =
        mlir::arith::AddIOp::create(builder, loc, indexValue, one);
    fir::StoreOp::create(builder, loc, newValue, index);
    return indexValue;
  }
  mlir::Value indexValue = index;
  index = mlir::arith::AddIOp::create(builder, loc, indexValue, one);
  return indexValue;
}

void fir::factory::Counter::reset(mlir::Location loc,
                                  fir::FirOpBuilder &builder) {
  if (canCountThroughLoops)
    fir::StoreOp::create(builder, loc, initialValue, index);
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
  temp = hlfir::DeclareOp::create(builder, loc, tempStorage, tempName, shape,
                                  lengths)
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
  hlfir::AssignOp::create(builder, loc, value, tempElement);
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
    fir::FreeMemOp::create(builder, loc, declare.getMemref());
  }
}

hlfir::Entity fir::factory::HomogeneousScalarStack::moveStackAsArrayExpr(
    mlir::Location loc, fir::FirOpBuilder &builder) {
  mlir::Value mustFree = builder.createBool(loc, allocateOnHeap);
  auto hlfirExpr = hlfir::AsExprOp::create(builder, loc, temp, mustFree);
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
    source = hlfir::Entity{hlfir::AsExprOp::create(builder, loc, source)};
  copy = hlfir::genAssociateExpr(loc, builder, source,
                                 source.getFortranElementType(), tempName);
}

void fir::factory::SimpleCopy::destroy(mlir::Location loc,
                                       fir::FirOpBuilder &builder) {
  hlfir::EndAssociateOp::create(builder, loc, copy);
}

//===----------------------------------------------------------------------===//
// fir::factory::AnyValueStack implementation.
//===----------------------------------------------------------------------===//

fir::factory::AnyValueStack::AnyValueStack(mlir::Location loc,
                                           fir::FirOpBuilder &builder,
                                           mlir::Type valueStaticType)
    : valueStaticType{valueStaticType},
      counter{loc, builder,
              builder.createIntegerConstant(loc, builder.getI64Type(), 0),
              /*stackThroughLoops=*/true} {
  opaquePtr = fir::runtime::genCreateValueStack(loc, builder);
  // Compute the storage type. I1 are stored as fir.logical<1>. This is required
  // to use descriptor.
  mlir::Type storageType =
      hlfir::getFortranElementOrSequenceType(valueStaticType);
  mlir::Type i1Type = builder.getI1Type();
  if (storageType == i1Type)
    storageType = fir::LogicalType::get(builder.getContext(), 1);
  assert(hlfir::getFortranElementType(storageType) != i1Type &&
         "array of i1 should not be used");
  mlir::Type heapType = fir::HeapType::get(storageType);
  mlir::Type boxType;
  if (hlfir::isPolymorphicType(valueStaticType))
    boxType = fir::ClassType::get(heapType);
  else
    boxType = fir::BoxType::get(heapType);
  retValueBox = builder.createTemporary(loc, boxType);
}

void fir::factory::AnyValueStack::pushValue(mlir::Location loc,
                                            fir::FirOpBuilder &builder,
                                            mlir::Value value) {
  hlfir::Entity entity{value};
  mlir::Type storageElementType =
      hlfir::getFortranElementType(retValueBox.getType());
  auto [box, maybeCleanUp] =
      hlfir::convertToBox(loc, builder, entity, storageElementType);
  fir::runtime::genPushValue(loc, builder, opaquePtr, fir::getBase(box));
  if (maybeCleanUp)
    (*maybeCleanUp)();
}

void fir::factory::AnyValueStack::resetFetchPosition(
    mlir::Location loc, fir::FirOpBuilder &builder) {
  counter.reset(loc, builder);
}

mlir::Value fir::factory::AnyValueStack::fetch(mlir::Location loc,
                                               fir::FirOpBuilder &builder) {
  mlir::Value indexValue = counter.getAndIncrementIndex(loc, builder);
  fir::runtime::genValueAt(loc, builder, opaquePtr, indexValue, retValueBox);
  // Dereference the allocatable "retValueBox", and load if trivial scalar
  // value.
  mlir::Value result =
      hlfir::loadTrivialScalar(loc, builder, hlfir::Entity{retValueBox});
  if (valueStaticType != result.getType()) {
    // Cast back saved simple scalars stored with another type to their original
    // type (like i1).
    if (fir::isa_trivial(valueStaticType))
      return builder.createConvert(loc, valueStaticType, result);
    // Memory type mismatches (e.g. fir.ref vs fir.heap) or hlfir.expr vs
    // variable type mismatches are OK, but the base Fortran type must be the
    // same.
    assert(hlfir::getFortranElementOrSequenceType(valueStaticType) ==
               hlfir::getFortranElementOrSequenceType(result.getType()) &&
           "non trivial values must be saved with their original type");
  }
  return result;
}

void fir::factory::AnyValueStack::destroy(mlir::Location loc,
                                          fir::FirOpBuilder &builder) {
  fir::runtime::genDestroyValueStack(loc, builder, opaquePtr);
}

//===----------------------------------------------------------------------===//
// fir::factory::AnyVariableStack implementation.
//===----------------------------------------------------------------------===//

fir::factory::AnyVariableStack::AnyVariableStack(mlir::Location loc,
                                                 fir::FirOpBuilder &builder,
                                                 mlir::Type variableStaticType)
    : variableStaticType{variableStaticType},
      counter{loc, builder,
              builder.createIntegerConstant(loc, builder.getI64Type(), 0),
              /*stackThroughLoops=*/true} {
  opaquePtr = fir::runtime::genCreateDescriptorStack(loc, builder);
  mlir::Type storageType =
      hlfir::getFortranElementOrSequenceType(variableStaticType);
  mlir::Type ptrType = fir::PointerType::get(storageType);
  mlir::Type boxType;
  if (hlfir::isPolymorphicType(variableStaticType))
    boxType = fir::ClassType::get(ptrType);
  else
    boxType = fir::BoxType::get(ptrType);
  retValueBox = builder.createTemporary(loc, boxType);
}

void fir::factory::AnyVariableStack::pushValue(mlir::Location loc,
                                               fir::FirOpBuilder &builder,
                                               mlir::Value variable) {
  hlfir::Entity entity{variable};
  mlir::Type storageElementType =
      hlfir::getFortranElementType(retValueBox.getType());
  auto [box, maybeCleanUp] =
      hlfir::convertToBox(loc, builder, entity, storageElementType);
  fir::runtime::genPushDescriptor(loc, builder, opaquePtr, fir::getBase(box));
  if (maybeCleanUp)
    (*maybeCleanUp)();
}

void fir::factory::AnyVariableStack::resetFetchPosition(
    mlir::Location loc, fir::FirOpBuilder &builder) {
  counter.reset(loc, builder);
}

mlir::Value fir::factory::AnyVariableStack::fetch(mlir::Location loc,
                                                  fir::FirOpBuilder &builder) {
  mlir::Value indexValue = counter.getAndIncrementIndex(loc, builder);
  fir::runtime::genDescriptorAt(loc, builder, opaquePtr, indexValue,
                                retValueBox);
  hlfir::Entity retBox{fir::LoadOp::create(builder, loc, retValueBox)};
  // The runtime always tracks variable as address, but the form of the variable
  // that was saved may be different (raw address, fir.boxchar), ensure
  // the returned variable has the same form of the one that was saved.
  if (mlir::isa<fir::BaseBoxType>(variableStaticType))
    return builder.createConvert(loc, variableStaticType, retBox);
  if (mlir::isa<fir::BoxCharType>(variableStaticType))
    return hlfir::genVariableBoxChar(loc, builder, retBox);
  mlir::Value rawAddr = genVariableRawAddress(loc, builder, retBox);
  return builder.createConvert(loc, variableStaticType, rawAddr);
}

void fir::factory::AnyVariableStack::destroy(mlir::Location loc,
                                             fir::FirOpBuilder &builder) {
  fir::runtime::genDestroyDescriptorStack(loc, builder, opaquePtr);
}

//===----------------------------------------------------------------------===//
// fir::factory::AnyVectorSubscriptStack implementation.
//===----------------------------------------------------------------------===//

fir::factory::AnyVectorSubscriptStack::AnyVectorSubscriptStack(
    mlir::Location loc, fir::FirOpBuilder &builder,
    mlir::Type variableStaticType, bool shapeCanBeSavedAsRegister, int rank)
    : AnyVariableStack{loc, builder, variableStaticType} {
  if (shapeCanBeSavedAsRegister) {
    shapeTemp = std::make_unique<TemporaryStorage>(SSARegister{});
    return;
  }
  // The shape will be tracked as the dimension inside a descriptor because
  // that is the easiest from a lowering point of view, and this is an
  // edge case situation that will probably not very well be exercised.
  mlir::Type type =
      fir::BoxType::get(builder.getVarLenSeqTy(builder.getI32Type(), rank));
  boxType = type;
  shapeTemp =
      std::make_unique<TemporaryStorage>(AnyVariableStack{loc, builder, type});
}

void fir::factory::AnyVectorSubscriptStack::pushShape(
    mlir::Location loc, fir::FirOpBuilder &builder, mlir::Value shape) {
  if (boxType) {
    // The shape is saved as a dimensions inside a descriptors.
    mlir::Type refType = fir::ReferenceType::get(
        hlfir::getFortranElementOrSequenceType(*boxType));
    mlir::Value null = builder.createNullConstant(loc, refType);
    mlir::Value descriptor =
        fir::EmboxOp::create(builder, loc, *boxType, null, shape);
    shapeTemp->pushValue(loc, builder, descriptor);
    return;
  }
  // Otherwise, simply keep track of the fir.shape itself, it is invariant.
  shapeTemp->cast<SSARegister>().pushValue(loc, builder, shape);
}

void fir::factory::AnyVectorSubscriptStack::resetFetchPosition(
    mlir::Location loc, fir::FirOpBuilder &builder) {
  static_cast<AnyVariableStack *>(this)->resetFetchPosition(loc, builder);
  shapeTemp->resetFetchPosition(loc, builder);
}

mlir::Value
fir::factory::AnyVectorSubscriptStack::fetchShape(mlir::Location loc,
                                                  fir::FirOpBuilder &builder) {
  if (boxType) {
    hlfir::Entity descriptor{shapeTemp->fetch(loc, builder)};
    return hlfir::genShape(loc, builder, descriptor);
  }
  return shapeTemp->cast<SSARegister>().fetch(loc, builder);
}

void fir::factory::AnyVectorSubscriptStack::destroy(
    mlir::Location loc, fir::FirOpBuilder &builder) {
  static_cast<AnyVariableStack *>(this)->destroy(loc, builder);
  shapeTemp->destroy(loc, builder);
}

//===----------------------------------------------------------------------===//
// fir::factory::AnyAddressStack implementation.
//===----------------------------------------------------------------------===//

fir::factory::AnyAddressStack::AnyAddressStack(mlir::Location loc,
                                               fir::FirOpBuilder &builder,
                                               mlir::Type addressType)
    : AnyValueStack(loc, builder, builder.getIntPtrType()),
      addressType{addressType} {}

void fir::factory::AnyAddressStack::pushValue(mlir::Location loc,
                                              fir::FirOpBuilder &builder,
                                              mlir::Value variable) {
  mlir::Value cast = variable;
  if (auto boxProcType = llvm::dyn_cast<fir::BoxProcType>(variable.getType())) {
    cast =
        fir::BoxAddrOp::create(builder, loc, boxProcType.getEleTy(), variable);
  }
  cast = builder.createConvert(loc, builder.getIntPtrType(), cast);
  static_cast<AnyValueStack *>(this)->pushValue(loc, builder, cast);
}

mlir::Value fir::factory::AnyAddressStack::fetch(mlir::Location loc,
                                                 fir::FirOpBuilder &builder) {
  mlir::Value addr = static_cast<AnyValueStack *>(this)->fetch(loc, builder);
  if (auto boxProcType = llvm::dyn_cast<fir::BoxProcType>(addressType)) {
    mlir::Value cast = builder.createConvert(loc, boxProcType.getEleTy(), addr);
    return fir::EmboxProcOp::create(builder, loc, boxProcType, cast);
  }
  return builder.createConvert(loc, addressType, addr);
}
