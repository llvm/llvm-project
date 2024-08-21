//===-- CIRGenBuilder.cpp - CIRBuilder implementation ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "CIRGenBuilder.h"

namespace cir {

mlir::Value CIRGenBuilderTy::maybeBuildArrayDecay(mlir::Location loc,
                                                  mlir::Value arrayPtr,
                                                  mlir::Type eltTy) {
  auto arrayPtrTy =
      ::mlir::dyn_cast<::mlir::cir::PointerType>(arrayPtr.getType());
  assert(arrayPtrTy && "expected pointer type");
  auto arrayTy =
      ::mlir::dyn_cast<::mlir::cir::ArrayType>(arrayPtrTy.getPointee());

  if (arrayTy) {
    mlir::cir::PointerType flatPtrTy =
        mlir::cir::PointerType::get(getContext(), arrayTy.getEltType());
    return create<mlir::cir::CastOp>(
        loc, flatPtrTy, mlir::cir::CastKind::array_to_ptrdecay, arrayPtr);
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
  return create<mlir::cir::PtrStrideOp>(arrayLocEnd, flatPtrTy, basePtr, idx);
}

mlir::cir::ConstantOp CIRGenBuilderTy::getConstInt(mlir::Location loc,
                                                   llvm::APSInt intVal) {
  bool isSigned = intVal.isSigned();
  auto width = intVal.getBitWidth();
  mlir::cir::IntType t = isSigned ? getSIntNTy(width) : getUIntNTy(width);
  return getConstInt(loc, t,
                     isSigned ? intVal.getSExtValue() : intVal.getZExtValue());
}

mlir::cir::ConstantOp CIRGenBuilderTy::getConstInt(mlir::Location loc,
                                                   llvm::APInt intVal) {
  auto width = intVal.getBitWidth();
  mlir::cir::IntType t = getUIntNTy(width);
  return getConstInt(loc, t, intVal.getZExtValue());
}

mlir::cir::ConstantOp CIRGenBuilderTy::getConstInt(mlir::Location loc,
                                                   mlir::Type t, uint64_t C) {
  auto intTy = mlir::dyn_cast<mlir::cir::IntType>(t);
  assert(intTy && "expected mlir::cir::IntType");
  return create<mlir::cir::ConstantOp>(loc, intTy,
                                       mlir::cir::IntAttr::get(t, C));
}
} // namespace cir
