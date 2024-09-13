//====- LoweringHelpers.cpp - Lowering helper functions -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains helper functions for lowering from CIR to LLVM or MLIR.
//
//===----------------------------------------------------------------------===//
#include "clang/CIR/LoweringHelpers.h"

mlir::DenseElementsAttr
convertStringAttrToDenseElementsAttr(mlir::cir::ConstArrayAttr attr,
                                     mlir::Type type) {
  auto values = llvm::SmallVector<mlir::APInt, 8>{};
  auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getElts());
  assert(stringAttr && "expected string attribute here");
  for (auto element : stringAttr)
    values.push_back({8, (uint64_t)element});
  auto arrayTy = mlir::dyn_cast<mlir::cir::ArrayType>(attr.getType());
  assert(arrayTy && "String attribute must have an array type");
  if (arrayTy.getSize() != stringAttr.size())
    llvm_unreachable("array type of the length not equal to that of the string "
                     "attribute is not supported yet");
  return mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get({(int64_t)values.size()}, type),
      llvm::ArrayRef(values));
}

template <> mlir::APInt getZeroInitFromType(mlir::Type Ty) {
  assert(mlir::isa<mlir::cir::IntType>(Ty) && "expected int type");
  auto IntTy = mlir::cast<mlir::cir::IntType>(Ty);
  return mlir::APInt::getZero(IntTy.getWidth());
}

template <> mlir::APFloat getZeroInitFromType(mlir::Type Ty) {
  assert((mlir::isa<mlir::cir::SingleType, mlir::cir::DoubleType>(Ty)) &&
         "only float and double supported");
  if (Ty.isF32() || mlir::isa<mlir::cir::SingleType>(Ty))
    return mlir::APFloat(0.f);
  if (Ty.isF64() || mlir::isa<mlir::cir::DoubleType>(Ty))
    return mlir::APFloat(0.0);
  llvm_unreachable("NYI");
}

// return the nested type and quantity of elements for cir.array type.
// e.g: for !cir.array<!cir.array<!s32i x 3> x 1>
// it returns !s32i as return value and stores 3 to elemQuantity.
mlir::Type getNestedTypeAndElemQuantity(mlir::Type Ty, unsigned &elemQuantity) {
  assert(mlir::isa<mlir::cir::ArrayType>(Ty) && "expected ArrayType");

  elemQuantity = 1;
  mlir::Type nestTy = Ty;
  while (auto ArrTy = mlir::dyn_cast<mlir::cir::ArrayType>(nestTy)) {
    nestTy = ArrTy.getEltType();
    elemQuantity *= ArrTy.getSize();
  }

  return nestTy;
}

template <typename AttrTy, typename StorageTy>
void convertToDenseElementsAttrImpl(mlir::cir::ConstArrayAttr attr,
                                    llvm::SmallVectorImpl<StorageTy> &values) {
  auto arrayAttr = mlir::cast<mlir::ArrayAttr>(attr.getElts());
  for (auto eltAttr : arrayAttr) {
    if (auto valueAttr = mlir::dyn_cast<AttrTy>(eltAttr)) {
      values.push_back(valueAttr.getValue());
    } else if (auto subArrayAttr =
                   mlir::dyn_cast<mlir::cir::ConstArrayAttr>(eltAttr)) {
      convertToDenseElementsAttrImpl<AttrTy>(subArrayAttr, values);
    } else if (auto zeroAttr = mlir::dyn_cast<mlir::cir::ZeroAttr>(eltAttr)) {
      unsigned numStoredZeros = 0;
      auto nestTy =
          getNestedTypeAndElemQuantity(zeroAttr.getType(), numStoredZeros);
      values.insert(values.end(), numStoredZeros,
                    getZeroInitFromType<StorageTy>(nestTy));
    } else {
      llvm_unreachable("unknown element in ConstArrayAttr");
    }
  }

  // Only fill in trailing zeros at the local cir.array level where the element
  // type isn't another array (for the mult-dim case).
  auto numTrailingZeros = attr.getTrailingZerosNum();
  if (numTrailingZeros) {
    auto localArrayTy = mlir::dyn_cast<mlir::cir::ArrayType>(attr.getType());
    assert(localArrayTy && "expected !cir.array");

    auto nestTy = localArrayTy.getEltType();
    if (!mlir::isa<mlir::cir::ArrayType>(nestTy))
      values.insert(values.end(), numTrailingZeros,
                    getZeroInitFromType<StorageTy>(nestTy));
  }
}

template <typename AttrTy, typename StorageTy>
mlir::DenseElementsAttr
convertToDenseElementsAttr(mlir::cir::ConstArrayAttr attr,
                           const llvm::SmallVectorImpl<int64_t> &dims,
                           mlir::Type type) {
  auto values = llvm::SmallVector<StorageTy, 8>{};
  convertToDenseElementsAttrImpl<AttrTy>(attr, values);
  return mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(dims, type),
                                      llvm::ArrayRef(values));
}

std::optional<mlir::Attribute>
lowerConstArrayAttr(mlir::cir::ConstArrayAttr constArr,
                    const mlir::TypeConverter *converter) {

  // Ensure ConstArrayAttr has a type.
  auto typedConstArr = mlir::dyn_cast<mlir::TypedAttr>(constArr);
  assert(typedConstArr && "cir::ConstArrayAttr is not a mlir::TypedAttr");

  // Ensure ConstArrayAttr type is a ArrayType.
  auto cirArrayType =
      mlir::dyn_cast<mlir::cir::ArrayType>(typedConstArr.getType());
  assert(cirArrayType && "cir::ConstArrayAttr is not a cir::ArrayType");

  // Is a ConstArrayAttr with an cir::ArrayType: fetch element type.
  mlir::Type type = cirArrayType;
  auto dims = llvm::SmallVector<int64_t, 2>{};
  while (auto arrayType = mlir::dyn_cast<mlir::cir::ArrayType>(type)) {
    dims.push_back(arrayType.getSize());
    type = arrayType.getEltType();
  }

  if (mlir::isa<mlir::StringAttr>(constArr.getElts()))
    return convertStringAttrToDenseElementsAttr(constArr,
                                                converter->convertType(type));
  if (mlir::isa<mlir::cir::IntType>(type))
    return convertToDenseElementsAttr<mlir::cir::IntAttr, mlir::APInt>(
        constArr, dims, converter->convertType(type));
  if (mlir::isa<mlir::cir::CIRFPTypeInterface>(type))
    return convertToDenseElementsAttr<mlir::cir::FPAttr, mlir::APFloat>(
        constArr, dims, converter->convertType(type));

  return std::nullopt;
}
