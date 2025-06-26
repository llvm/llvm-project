//===----------------------------------------------------------------------===//
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
  const auto arrayPtrTy = mlir::cast<cir::PointerType>(arrayPtr.getType());
  const auto arrayTy = mlir::dyn_cast<cir::ArrayType>(arrayPtrTy.getPointee());

  if (arrayTy) {
    const cir::PointerType flatPtrTy = getPointerTo(arrayTy.getElementType());
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
  const mlir::Type flatPtrTy = basePtr.getType();
  return create<cir::PtrStrideOp>(arrayLocEnd, flatPtrTy, basePtr, idx);
}

cir::ConstantOp CIRGenBuilderTy::getConstInt(mlir::Location loc,
                                             llvm::APSInt intVal) {
  bool isSigned = intVal.isSigned();
  unsigned width = intVal.getBitWidth();
  cir::IntType t = isSigned ? getSIntNTy(width) : getUIntNTy(width);
  return getConstInt(loc, t,
                     isSigned ? intVal.getSExtValue() : intVal.getZExtValue());
}

cir::ConstantOp CIRGenBuilderTy::getConstInt(mlir::Location loc,
                                             llvm::APInt intVal) {
  return getConstInt(loc, llvm::APSInt(intVal));
}

cir::ConstantOp CIRGenBuilderTy::getConstInt(mlir::Location loc, mlir::Type t,
                                             uint64_t c) {
  assert(mlir::isa<cir::IntType>(t) && "expected cir::IntType");
  return create<cir::ConstantOp>(loc, cir::IntAttr::get(t, c));
}

cir::ConstantOp
clang::CIRGen::CIRGenBuilderTy::getConstFP(mlir::Location loc, mlir::Type t,
                                           llvm::APFloat fpVal) {
  assert(mlir::isa<cir::CIRFPTypeInterface>(t) &&
         "expected floating point type");
  return create<cir::ConstantOp>(loc, getAttr<cir::FPAttr>(t, fpVal));
}

// This can't be defined in Address.h because that file is included by
// CIRGenBuilder.h
Address Address::withElementType(CIRGenBuilderTy &builder,
                                 mlir::Type elemTy) const {
  assert(!cir::MissingFeatures::addressOffset());
  assert(!cir::MissingFeatures::addressIsKnownNonNull());
  assert(!cir::MissingFeatures::addressPointerAuthInfo());

  return Address(builder.createPtrBitcast(getBasePointer(), elemTy), elemTy,
                 getAlignment());
}
