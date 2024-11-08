//===-- CIRGenBuilder.cpp - CIRBuilder implementation ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "CIRGenBuilder.h"

using namespace clang::CIRGen;

mlir::Value CIRGenBuilderTy::maybeBuildArrayDecay(mlir::Location loc,
                                                  mlir::Value arrayPtr,
                                                  mlir::Type eltTy) {
  auto arrayPtrTy = ::mlir::dyn_cast<cir::PointerType>(arrayPtr.getType());
  assert(arrayPtrTy && "expected pointer type");
  auto arrayTy = ::mlir::dyn_cast<cir::ArrayType>(arrayPtrTy.getPointee());

  if (arrayTy) {
    auto addrSpace = ::mlir::cast_if_present<cir::AddressSpaceAttr>(
        arrayPtrTy.getAddrSpace());
    cir::PointerType flatPtrTy = getPointerTo(arrayTy.getEltType(), addrSpace);
    return create<cir::CastOp>(loc, flatPtrTy, cir::CastKind::array_to_ptrdecay,
                               arrayPtr);
  }

  assert(arrayPtrTy.getPointee() == eltTy &&
         "flat pointee type must match original array element type");
  return arrayPtr;
}

mlir::Value CIRGenBuilderTy::getArrayElement(mlir::Location arrayLocBegin,
                                             mlir::Location arrayLocEnd,
                                             mlir::Value arrayPtr,
                                             mlir::Type eltTy, mlir::Value idx,
                                             bool shouldDecay) {
  mlir::Value basePtr = arrayPtr;
  if (shouldDecay)
    basePtr = maybeBuildArrayDecay(arrayLocBegin, arrayPtr, eltTy);
  mlir::Type flatPtrTy = basePtr.getType();
  return create<cir::PtrStrideOp>(arrayLocEnd, flatPtrTy, basePtr, idx);
}

cir::ConstantOp CIRGenBuilderTy::getConstInt(mlir::Location loc,
                                             llvm::APSInt intVal) {
  bool isSigned = intVal.isSigned();
  auto width = intVal.getBitWidth();
  cir::IntType t = isSigned ? getSIntNTy(width) : getUIntNTy(width);
  return getConstInt(loc, t,
                     isSigned ? intVal.getSExtValue() : intVal.getZExtValue());
}

cir::ConstantOp CIRGenBuilderTy::getConstInt(mlir::Location loc,
                                             llvm::APInt intVal) {
  auto width = intVal.getBitWidth();
  cir::IntType t = getUIntNTy(width);
  return getConstInt(loc, t, intVal.getZExtValue());
}

cir::ConstantOp CIRGenBuilderTy::getConstInt(mlir::Location loc, mlir::Type t,
                                             uint64_t C) {
  auto intTy = mlir::dyn_cast<cir::IntType>(t);
  assert(intTy && "expected cir::IntType");
  return create<cir::ConstantOp>(loc, intTy, cir::IntAttr::get(t, C));
}
