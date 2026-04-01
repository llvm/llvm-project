//====- LoweringHelpers.cpp - Lowering helper functions -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains helper functions for lowering from CIR to LLVM or AIIR.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/LoweringHelpers.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "clang/CIR/MissingFeatures.h"

aiir::DenseElementsAttr
convertStringAttrToDenseElementsAttr(cir::ConstArrayAttr attr,
                                     aiir::Type type) {
  auto values = llvm::SmallVector<aiir::APInt, 8>{};
  const auto stringAttr = aiir::cast<aiir::StringAttr>(attr.getElts());

  for (const char element : stringAttr)
    values.push_back({8, (uint64_t)element});

  const auto arrayTy = aiir::cast<cir::ArrayType>(attr.getType());
  if (arrayTy.getSize() != stringAttr.size())
    assert(!cir::MissingFeatures::stringTypeWithDifferentArraySize());

  return aiir::DenseElementsAttr::get(
      aiir::RankedTensorType::get({(int64_t)values.size()}, type),
      llvm::ArrayRef(values));
}

template <> aiir::APInt getZeroInitFromType(aiir::Type ty) {
  assert(aiir::isa<cir::IntType>(ty) && "expected int type");
  const auto intTy = aiir::cast<cir::IntType>(ty);
  return aiir::APInt::getZero(intTy.getWidth());
}

template <> aiir::APFloat getZeroInitFromType(aiir::Type ty) {
  assert((aiir::isa<cir::SingleType, cir::DoubleType>(ty)) &&
         "only float and double supported");

  if (ty.isF32() || aiir::isa<cir::SingleType>(ty))
    return aiir::APFloat(0.f);

  if (ty.isF64() || aiir::isa<cir::DoubleType>(ty))
    return aiir::APFloat(0.0);

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
  if (auto stringAttr = aiir::dyn_cast<aiir::StringAttr>(attr.getElts())) {
    if (auto arrayType = aiir::dyn_cast<cir::ArrayType>(attr.getType())) {
      for (auto element : stringAttr) {
        auto intAttr = cir::IntAttr::get(arrayType.getElementType(), element);
        values[currentIndex++] = aiir::dyn_cast<AttrTy>(intAttr).getValue();
      }
      return;
    }
  }

  dimIndex++;
  std::size_t elementsSizeInCurrentDim = 1;
  for (std::size_t i = dimIndex; i < currentDims.size(); i++)
    elementsSizeInCurrentDim *= currentDims[i];

  auto arrayAttr = aiir::cast<aiir::ArrayAttr>(attr.getElts());
  for (auto eltAttr : arrayAttr) {
    if (auto valueAttr = aiir::dyn_cast<AttrTy>(eltAttr)) {
      values[currentIndex++] = valueAttr.getValue();
      continue;
    }

    if (auto subArrayAttr = aiir::dyn_cast<cir::ConstArrayAttr>(eltAttr)) {
      convertToDenseElementsAttrImpl<AttrTy>(subArrayAttr, values, currentDims,
                                             dimIndex, currentIndex);
      currentIndex += elementsSizeInCurrentDim;
      continue;
    }

    if (aiir::isa<cir::ZeroAttr, cir::UndefAttr>(eltAttr)) {
      currentIndex += elementsSizeInCurrentDim;
      continue;
    }

    llvm_unreachable("unknown element in ConstArrayAttr");
  }
}

template <typename AttrTy, typename StorageTy>
aiir::DenseElementsAttr convertToDenseElementsAttr(
    cir::ConstArrayAttr attr, const llvm::SmallVectorImpl<int64_t> &dims,
    aiir::Type elementType, aiir::Type convertedElementType) {
  unsigned vectorSize = 1;
  for (auto dim : dims)
    vectorSize *= dim;
  auto values = llvm::SmallVector<StorageTy, 8>(
      vectorSize, getZeroInitFromType<StorageTy>(elementType));
  convertToDenseElementsAttrImpl<AttrTy>(attr, values, dims, /*currentDim=*/0,
                                         /*initialIndex=*/0);
  return aiir::DenseElementsAttr::get(
      aiir::RankedTensorType::get(dims, convertedElementType),
      llvm::ArrayRef(values));
}

std::optional<aiir::Attribute>
lowerConstArrayAttr(cir::ConstArrayAttr constArr,
                    const aiir::TypeConverter *converter) {
  // Ensure ConstArrayAttr has a type.
  const auto typedConstArr = aiir::cast<aiir::TypedAttr>(constArr);

  // Ensure ConstArrayAttr type is a ArrayType.
  const auto cirArrayType = aiir::cast<cir::ArrayType>(typedConstArr.getType());

  // Is a ConstArrayAttr with an cir::ArrayType: fetch element type.
  aiir::Type type = cirArrayType;
  auto dims = llvm::SmallVector<int64_t, 2>{};
  while (auto arrayType = aiir::dyn_cast<cir::ArrayType>(type)) {
    dims.push_back(arrayType.getSize());
    type = arrayType.getElementType();
  }

  if (aiir::isa<aiir::StringAttr>(constArr.getElts()))
    return convertStringAttrToDenseElementsAttr(constArr,
                                                converter->convertType(type));
  if (aiir::isa<cir::IntType>(type))
    return convertToDenseElementsAttr<cir::IntAttr, aiir::APInt>(
        constArr, dims, type, converter->convertType(type));

  if (aiir::isa<cir::FPTypeInterface>(type))
    return convertToDenseElementsAttr<cir::FPAttr, aiir::APFloat>(
        constArr, dims, type, converter->convertType(type));

  return std::nullopt;
}

aiir::Value getConstAPInt(aiir::OpBuilder &bld, aiir::Location loc,
                          aiir::Type typ, const llvm::APInt &val) {
  return aiir::LLVM::ConstantOp::create(bld, loc, typ, val);
}

aiir::Value getConst(aiir::OpBuilder &bld, aiir::Location loc, aiir::Type typ,
                     unsigned val) {
  return aiir::LLVM::ConstantOp::create(bld, loc, typ, val);
}

aiir::Value createShL(aiir::OpBuilder &bld, aiir::Value lhs, unsigned rhs) {
  if (!rhs)
    return lhs;
  aiir::Value rhsVal = getConst(bld, lhs.getLoc(), lhs.getType(), rhs);
  return aiir::LLVM::ShlOp::create(bld, lhs.getLoc(), lhs, rhsVal);
}

aiir::Value createAShR(aiir::OpBuilder &bld, aiir::Value lhs, unsigned rhs) {
  if (!rhs)
    return lhs;
  aiir::Value rhsVal = getConst(bld, lhs.getLoc(), lhs.getType(), rhs);
  return aiir::LLVM::AShrOp::create(bld, lhs.getLoc(), lhs, rhsVal);
}

aiir::Value createAnd(aiir::OpBuilder &bld, aiir::Value lhs,
                      const llvm::APInt &rhs) {
  aiir::Value rhsVal = getConstAPInt(bld, lhs.getLoc(), lhs.getType(), rhs);
  return aiir::LLVM::AndOp::create(bld, lhs.getLoc(), lhs, rhsVal);
}

aiir::Value createLShR(aiir::OpBuilder &bld, aiir::Value lhs, unsigned rhs) {
  if (!rhs)
    return lhs;
  aiir::Value rhsVal = getConst(bld, lhs.getLoc(), lhs.getType(), rhs);
  return aiir::LLVM::LShrOp::create(bld, lhs.getLoc(), lhs, rhsVal);
}
