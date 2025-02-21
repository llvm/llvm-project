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

void CIRGenBuilderTy::computeGlobalViewIndicesFromFlatOffset(
    int64_t Offset, mlir::Type Ty, cir::CIRDataLayout Layout,
    llvm::SmallVectorImpl<int64_t> &Indices) {
  if (!Offset)
    return;

  mlir::Type SubType;

  auto getIndexAndNewOffset =
      [](int64_t Offset, int64_t EltSize) -> std::pair<int64_t, int64_t> {
    int64_t DivRet = Offset / EltSize;
    if (DivRet < 0)
      DivRet -= 1; // make sure offset is positive
    int64_t ModRet = Offset - (DivRet * EltSize);
    return {DivRet, ModRet};
  };

  if (auto ArrayTy = mlir::dyn_cast<cir::ArrayType>(Ty)) {
    int64_t EltSize = Layout.getTypeAllocSize(ArrayTy.getEltType());
    SubType = ArrayTy.getEltType();
    const auto [Index, NewOffset] = getIndexAndNewOffset(Offset, EltSize);
    Indices.push_back(Index);
    Offset = NewOffset;
  } else if (auto StructTy = mlir::dyn_cast<cir::StructType>(Ty)) {
    auto Elts = StructTy.getMembers();
    int64_t Pos = 0;
    for (size_t I = 0; I < Elts.size(); ++I) {
      int64_t EltSize =
          (int64_t)Layout.getTypeAllocSize(Elts[I]).getFixedValue();
      unsigned AlignMask = Layout.getABITypeAlign(Elts[I]).value() - 1;
      if (StructTy.getPacked())
        AlignMask = 0;
      Pos = (Pos + AlignMask) & ~AlignMask;
      assert(Offset >= 0);
      if (Offset < Pos + EltSize) {
        Indices.push_back(I);
        SubType = Elts[I];
        Offset -= Pos;
        break;
      }
      Pos += EltSize;
    }
  } else {
    llvm_unreachable("unexpected type");
  }

  assert(SubType);
  computeGlobalViewIndicesFromFlatOffset(Offset, SubType, Layout, Indices);
}

uint64_t CIRGenBuilderTy::computeOffsetFromGlobalViewIndices(
    const cir::CIRDataLayout &layout, mlir::Type typ,
    llvm::ArrayRef<int64_t> indexes) {

  int64_t offset = 0;
  for (auto idx : indexes) {
    if (auto sTy = dyn_cast<cir::StructType>(typ)) {
      offset += sTy.getElementOffset(layout.layout, idx);
      assert(idx < sTy.getMembers().size());
      typ = sTy.getMembers()[idx];
    } else if (auto arTy = dyn_cast<cir::ArrayType>(typ)) {
      typ = arTy.getEltType();
      offset += layout.getTypeAllocSize(typ) * idx;
    } else {
      llvm_unreachable("NYI");
    }
  }

  return offset;
}

// This can't be defined in Address.h because that file is included by
// CIRGenBuilder.h
Address Address::withElementType(CIRGenBuilderTy &builder,
                                 mlir::Type ElemTy) const {
  if (!hasOffset())
    return Address(builder.createPtrBitcast(getBasePointer(), ElemTy), ElemTy,
                   getAlignment(), getPointerAuthInfo(), /*Offset=*/nullptr,
                   isKnownNonNull());
  return Address(builder.createPtrBitcast(getPointer(), ElemTy), ElemTy,
                 getAlignment(), isKnownNonNull());
}
