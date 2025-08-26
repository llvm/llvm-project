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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "clang/CIR/MissingFeatures.h"

mlir::DenseElementsAttr
convertStringAttrToDenseElementsAttr(cir::ConstArrayAttr attr,
                                     mlir::Type type) {
  auto values = llvm::SmallVector<mlir::APInt, 8>{};
  const auto stringAttr = mlir::cast<mlir::StringAttr>(attr.getElts());

  for (const char element : stringAttr)
    values.push_back({8, (uint64_t)element});

  const auto arrayTy = mlir::cast<cir::ArrayType>(attr.getType());
  if (arrayTy.getSize() != stringAttr.size())
    assert(!cir::MissingFeatures::stringTypeWithDifferentArraySize());

  return mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get({(int64_t)values.size()}, type),
      llvm::ArrayRef(values));
}

template <> mlir::APInt getZeroInitFromType(mlir::Type ty) {
  assert(mlir::isa<cir::IntType>(ty) && "expected int type");
  const auto intTy = mlir::cast<cir::IntType>(ty);
  return mlir::APInt::getZero(intTy.getWidth());
}

template <> mlir::APFloat getZeroInitFromType(mlir::Type ty) {
  assert((mlir::isa<cir::SingleType, cir::DoubleType>(ty)) &&
         "only float and double supported");

  if (ty.isF32() || mlir::isa<cir::SingleType>(ty))
    return mlir::APFloat(0.f);

  if (ty.isF64() || mlir::isa<cir::DoubleType>(ty))
    return mlir::APFloat(0.0);

  llvm_unreachable("NYI");
}

/// \param attr the ConstArrayAttr to convert
/// \param values the output parameter, the values array to fill
/// \param currentDims the shpae of tensor we're going to convert to
/// \param dimIndex the current dimension we're processing
/// \param currentIndex the current index in the values array
template <typename AttrTy, typename StorageTy>
void convertToDenseElementsAttrImpl(
    cir::ConstArrayAttr attr, llvm::SmallVectorImpl<StorageTy> &values,
    const llvm::SmallVectorImpl<int64_t> &currentDims, int64_t dimIndex,
    int64_t currentIndex) {
  if (auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getElts())) {
    if (auto arrayType = mlir::dyn_cast<cir::ArrayType>(attr.getType())) {
      for (auto element : stringAttr) {
        auto intAttr = cir::IntAttr::get(arrayType.getElementType(), element);
        values[currentIndex++] = mlir::dyn_cast<AttrTy>(intAttr).getValue();
      }
      return;
    }
  }

  dimIndex++;
  std::size_t elementsSizeInCurrentDim = 1;
  for (std::size_t i = dimIndex; i < currentDims.size(); i++)
    elementsSizeInCurrentDim *= currentDims[i];

  auto arrayAttr = mlir::cast<mlir::ArrayAttr>(attr.getElts());
  for (auto eltAttr : arrayAttr) {
    if (auto valueAttr = mlir::dyn_cast<AttrTy>(eltAttr)) {
      values[currentIndex++] = valueAttr.getValue();
      continue;
    }

    if (auto subArrayAttr = mlir::dyn_cast<cir::ConstArrayAttr>(eltAttr)) {
      convertToDenseElementsAttrImpl<AttrTy>(subArrayAttr, values, currentDims,
                                             dimIndex, currentIndex);
      currentIndex += elementsSizeInCurrentDim;
      continue;
    }

    if (mlir::isa<cir::ZeroAttr, cir::UndefAttr>(eltAttr)) {
      currentIndex += elementsSizeInCurrentDim;
      continue;
    }

    llvm_unreachable("unknown element in ConstArrayAttr");
  }
}

template <typename AttrTy, typename StorageTy>
mlir::DenseElementsAttr convertToDenseElementsAttr(
    cir::ConstArrayAttr attr, const llvm::SmallVectorImpl<int64_t> &dims,
    mlir::Type elementType, mlir::Type convertedElementType) {
  unsigned vectorSize = 1;
  for (auto dim : dims)
    vectorSize *= dim;
  auto values = llvm::SmallVector<StorageTy, 8>(
      vectorSize, getZeroInitFromType<StorageTy>(elementType));
  convertToDenseElementsAttrImpl<AttrTy>(attr, values, dims, /*currentDim=*/0,
                                         /*initialIndex=*/0);
  return mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get(dims, convertedElementType),
      llvm::ArrayRef(values));
}

std::optional<mlir::Attribute>
lowerConstArrayAttr(cir::ConstArrayAttr constArr,
                    const mlir::TypeConverter *converter) {
  // Ensure ConstArrayAttr has a type.
  const auto typedConstArr = mlir::cast<mlir::TypedAttr>(constArr);

  // Ensure ConstArrayAttr type is a ArrayType.
  const auto cirArrayType = mlir::cast<cir::ArrayType>(typedConstArr.getType());

  // Is a ConstArrayAttr with an cir::ArrayType: fetch element type.
  mlir::Type type = cirArrayType;
  auto dims = llvm::SmallVector<int64_t, 2>{};
  while (auto arrayType = mlir::dyn_cast<cir::ArrayType>(type)) {
    dims.push_back(arrayType.getSize());
    type = arrayType.getElementType();
  }

  if (mlir::isa<mlir::StringAttr>(constArr.getElts()))
    return convertStringAttrToDenseElementsAttr(constArr,
                                                converter->convertType(type));
  if (mlir::isa<cir::IntType>(type))
    return convertToDenseElementsAttr<cir::IntAttr, mlir::APInt>(
        constArr, dims, type, converter->convertType(type));

  if (mlir::isa<cir::FPTypeInterface>(type))
    return convertToDenseElementsAttr<cir::FPAttr, mlir::APFloat>(
        constArr, dims, type, converter->convertType(type));

  return std::nullopt;
}

mlir::Value getConstAPInt(mlir::OpBuilder &bld, mlir::Location loc,
                          mlir::Type typ, const llvm::APInt &val) {
  return bld.create<mlir::LLVM::ConstantOp>(loc, typ, val);
}

mlir::Value getConst(mlir::OpBuilder &bld, mlir::Location loc, mlir::Type typ,
                     unsigned val) {
  return bld.create<mlir::LLVM::ConstantOp>(loc, typ, val);
}

mlir::Value createShL(mlir::OpBuilder &bld, mlir::Value lhs, unsigned rhs) {
  if (!rhs)
    return lhs;
  mlir::Value rhsVal = getConst(bld, lhs.getLoc(), lhs.getType(), rhs);
  return bld.create<mlir::LLVM::ShlOp>(lhs.getLoc(), lhs, rhsVal);
}

mlir::Value createAShR(mlir::OpBuilder &bld, mlir::Value lhs, unsigned rhs) {
  if (!rhs)
    return lhs;
  mlir::Value rhsVal = getConst(bld, lhs.getLoc(), lhs.getType(), rhs);
  return bld.create<mlir::LLVM::AShrOp>(lhs.getLoc(), lhs, rhsVal);
}

mlir::Value createAnd(mlir::OpBuilder &bld, mlir::Value lhs,
                      const llvm::APInt &rhs) {
  mlir::Value rhsVal = getConstAPInt(bld, lhs.getLoc(), lhs.getType(), rhs);
  return bld.create<mlir::LLVM::AndOp>(lhs.getLoc(), lhs, rhsVal);
}

mlir::Value createLShR(mlir::OpBuilder &bld, mlir::Value lhs, unsigned rhs) {
  if (!rhs)
    return lhs;
  mlir::Value rhsVal = getConst(bld, lhs.getLoc(), lhs.getType(), rhs);
  return bld.create<mlir::LLVM::LShrOp>(lhs.getLoc(), lhs, rhsVal);
}
