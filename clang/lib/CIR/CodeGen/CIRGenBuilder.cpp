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
  assert(mlir::isa<cir::FPTypeInterface>(t) && "expected floating point type");
  return create<cir::ConstantOp>(loc, cir::FPAttr::get(t, fpVal));
}

void CIRGenBuilderTy::computeGlobalViewIndicesFromFlatOffset(
    int64_t offset, mlir::Type ty, cir::CIRDataLayout layout,
    llvm::SmallVectorImpl<int64_t> &indices) {
  if (!offset)
    return;

  mlir::Type subType;

  auto getIndexAndNewOffset =
      [](int64_t offset, int64_t eltSize) -> std::pair<int64_t, int64_t> {
    int64_t divRet = offset / eltSize;
    if (divRet < 0)
      divRet -= 1; // make sure offset is positive
    int64_t modRet = offset - (divRet * eltSize);
    return {divRet, modRet};
  };

  if (auto arrayTy = mlir::dyn_cast<cir::ArrayType>(ty)) {
    int64_t eltSize = layout.getTypeAllocSize(arrayTy.getElementType());
    subType = arrayTy.getElementType();
    const auto [index, newOffset] = getIndexAndNewOffset(offset, eltSize);
    indices.push_back(index);
    offset = newOffset;
  } else if (auto recordTy = mlir::dyn_cast<cir::RecordType>(ty)) {
    auto elts = recordTy.getMembers();
    int64_t Pos = 0;
    for (size_t i = 0; i < elts.size(); ++i) {
      int64_t eltSize =
          (int64_t)layout.getTypeAllocSize(elts[i]).getFixedValue();
      unsigned alignMask = layout.getABITypeAlign(elts[i]).value() - 1;
      if (recordTy.getPacked())
        alignMask = 0;
      // Union's fields have the same offset, so no need to change Pos here,
      // we just need to find EltSize that is greater then the required offset.
      // The same is true for the similar union type check below
      if (!recordTy.isUnion())
        Pos = (Pos + alignMask) & ~alignMask;
      assert(offset >= 0);
      if (offset < Pos + eltSize) {
        indices.push_back(i);
        subType = elts[i];
        offset -= Pos;
        break;
      }
      // No need to update Pos here, see the comment above.
      if (!recordTy.isUnion())
        Pos += eltSize;
    }
  } else {
    llvm_unreachable("unexpected type");
  }

  assert(subType);
  computeGlobalViewIndicesFromFlatOffset(offset, subType, layout, indices);
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
