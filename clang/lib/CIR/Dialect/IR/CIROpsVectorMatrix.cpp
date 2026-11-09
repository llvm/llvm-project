//===- CIROpsVectorMatrix.cpp - CIR ops in partition `VectorMatrix` -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CIR ops in partition `VectorMatrix`: the
// hand-written parts first, then the TableGen-generated definitions.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIROpsVectorMatrix.h"
#include "clang/CIR/Dialect/IR/CIROpsCore.h"

#include "CIROpDefinitions.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace cir;

//===----------------------------------------------------------------------===//
// VecCreateOp
//===----------------------------------------------------------------------===//

OpFoldResult cir::VecCreateOp::fold(FoldAdaptor adaptor) {
  if (llvm::any_of(getElements(), [](mlir::Value value) {
        return !value.getDefiningOp<cir::ConstantOp>();
      }))
    return {};

  return cir::ConstVectorAttr::get(
      getType(), mlir::ArrayAttr::get(getContext(), adaptor.getElements()));
}

LogicalResult cir::VecCreateOp::verify() {
  // Verify that the number of arguments matches the number of elements in the
  // vector, and that the type of all the arguments matches the type of the
  // elements in the vector.
  const cir::VectorType vecTy = getType();
  if (vecTy.isScalable())
    return emitOpError() << "scalable vectors are not supported";

  if (getElements().size() != vecTy.getFixedSize()) {
    return emitOpError() << "operand count of " << getElements().size()
                         << " doesn't match vector type " << vecTy
                         << " element count of " << vecTy.getFixedSize();
  }

  const mlir::Type elementType = vecTy.getElementType();
  for (const mlir::Value element : getElements()) {
    if (element.getType() != elementType) {
      return emitOpError() << "operand type " << element.getType()
                           << " doesn't match vector element type "
                           << elementType;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// VecExtractOp
//===----------------------------------------------------------------------===//

OpFoldResult cir::VecExtractOp::fold(FoldAdaptor adaptor) {
  const auto vectorAttr =
      llvm::dyn_cast_if_present<cir::ConstVectorAttr>(adaptor.getVec());
  if (!vectorAttr)
    return {};

  const auto indexAttr =
      llvm::dyn_cast_if_present<cir::IntAttr>(adaptor.getIndex());
  if (!indexAttr)
    return {};

  const mlir::ArrayAttr elements = vectorAttr.getElts();
  const uint64_t index = indexAttr.getUInt();
  if (index >= elements.size())
    return {};

  return elements[index];
}

//===----------------------------------------------------------------------===//
// VecCmpOp
//===----------------------------------------------------------------------===//

LogicalResult cir::VecCmpOp::verify() {
  if (getFenvAttr() && !cir::isFPOrVectorOfFPType(getLhs().getType()))
    return emitOpError()
           << "'fenv' is only valid for floating-point comparisons";
  return success();
}

OpFoldResult cir::VecCmpOp::fold(FoldAdaptor adaptor) {
  // Do not fold when fenv is present.
  // TODO(cir): This is overly conservative. We could fold comparisons as long
  //            as we can prove that the operation wouldn't raise exceptions or
  //            when the fenv attribute does not require strict exception
  //            semantics.
  if (getFenvAttr())
    return {};

  auto lhsVecAttr =
      mlir::dyn_cast_if_present<cir::ConstVectorAttr>(adaptor.getLhs());
  auto rhsVecAttr =
      mlir::dyn_cast_if_present<cir::ConstVectorAttr>(adaptor.getRhs());
  if (!lhsVecAttr || !rhsVecAttr)
    return {};

  mlir::Type inputElemTy =
      mlir::cast<cir::VectorType>(lhsVecAttr.getType()).getElementType();
  if (!isAnyIntegerOrFloatingPointType(inputElemTy))
    return {};

  cir::CmpOpKind opKind = adaptor.getKind();
  mlir::ArrayAttr lhsVecElhs = lhsVecAttr.getElts();
  mlir::ArrayAttr rhsVecElhs = rhsVecAttr.getElts();
  uint64_t vecSize = lhsVecElhs.size();

  SmallVector<mlir::Attribute, 16> elements(vecSize);
  bool isIntAttr = vecSize && mlir::isa<cir::IntAttr>(lhsVecElhs[0]);
  bool isUnsignedInt =
      isIntAttr && mlir::cast<cir::IntType>(inputElemTy).isUnsigned();
  for (uint64_t i = 0; i < vecSize; i++) {
    mlir::Attribute lhsAttr = lhsVecElhs[i];
    mlir::Attribute rhsAttr = rhsVecElhs[i];
    bool cmpResult = false;
    switch (opKind) {
    case cir::CmpOpKind::lt: {
      if (isIntAttr) {
        if (isUnsignedInt)
          cmpResult = mlir::cast<cir::IntAttr>(lhsAttr).getUInt() <
                      mlir::cast<cir::IntAttr>(rhsAttr).getUInt();
        else
          cmpResult = mlir::cast<cir::IntAttr>(lhsAttr).getSInt() <
                      mlir::cast<cir::IntAttr>(rhsAttr).getSInt();
      } else {
        cmpResult = mlir::cast<cir::FPAttr>(lhsAttr).getValue() <
                    mlir::cast<cir::FPAttr>(rhsAttr).getValue();
      }
      break;
    }
    case cir::CmpOpKind::le: {
      if (isIntAttr) {
        if (isUnsignedInt)
          cmpResult = mlir::cast<cir::IntAttr>(lhsAttr).getUInt() <=
                      mlir::cast<cir::IntAttr>(rhsAttr).getUInt();
        else
          cmpResult = mlir::cast<cir::IntAttr>(lhsAttr).getSInt() <=
                      mlir::cast<cir::IntAttr>(rhsAttr).getSInt();
      } else {
        cmpResult = mlir::cast<cir::FPAttr>(lhsAttr).getValue() <=
                    mlir::cast<cir::FPAttr>(rhsAttr).getValue();
      }
      break;
    }
    case cir::CmpOpKind::gt: {
      if (isIntAttr) {
        if (isUnsignedInt)
          cmpResult = mlir::cast<cir::IntAttr>(lhsAttr).getUInt() >
                      mlir::cast<cir::IntAttr>(rhsAttr).getUInt();
        else
          cmpResult = mlir::cast<cir::IntAttr>(lhsAttr).getSInt() >
                      mlir::cast<cir::IntAttr>(rhsAttr).getSInt();
      } else {
        cmpResult = mlir::cast<cir::FPAttr>(lhsAttr).getValue() >
                    mlir::cast<cir::FPAttr>(rhsAttr).getValue();
      }
      break;
    }
    case cir::CmpOpKind::ge: {
      if (isIntAttr) {
        if (isUnsignedInt)
          cmpResult = mlir::cast<cir::IntAttr>(lhsAttr).getUInt() >=
                      mlir::cast<cir::IntAttr>(rhsAttr).getUInt();
        else
          cmpResult = mlir::cast<cir::IntAttr>(lhsAttr).getSInt() >=
                      mlir::cast<cir::IntAttr>(rhsAttr).getSInt();
      } else {
        cmpResult = mlir::cast<cir::FPAttr>(lhsAttr).getValue() >=
                    mlir::cast<cir::FPAttr>(rhsAttr).getValue();
      }
      break;
    }
    case cir::CmpOpKind::eq: {
      if (isIntAttr) {
        cmpResult = mlir::cast<cir::IntAttr>(lhsAttr).getSInt() ==
                    mlir::cast<cir::IntAttr>(rhsAttr).getSInt();
      } else {
        cmpResult = mlir::cast<cir::FPAttr>(lhsAttr).getValue() ==
                    mlir::cast<cir::FPAttr>(rhsAttr).getValue();
      }
      break;
    }
    case cir::CmpOpKind::ne: {
      if (isIntAttr) {
        cmpResult = mlir::cast<cir::IntAttr>(lhsAttr).getSInt() !=
                    mlir::cast<cir::IntAttr>(rhsAttr).getSInt();
      } else {
        cmpResult = mlir::cast<cir::FPAttr>(lhsAttr).getValue() !=
                    mlir::cast<cir::FPAttr>(rhsAttr).getValue();
      }
      break;
    }
    case cir::CmpOpKind::one: {
      llvm::APFloat::cmpResult cr =
          mlir::cast<cir::FPAttr>(lhsAttr).getValue().compare(
              mlir::cast<cir::FPAttr>(rhsAttr).getValue());
      cmpResult =
          cr != llvm::APFloat::cmpUnordered && cr != llvm::APFloat::cmpEqual;
      break;
    }
    case cir::CmpOpKind::uno: {
      cmpResult = mlir::cast<cir::FPAttr>(lhsAttr).getValue().compare(
                      mlir::cast<cir::FPAttr>(rhsAttr).getValue()) ==
                  llvm::APFloat::cmpUnordered;
      break;
    }
    }

    // A true result is all bits set (-1 in two's complement), and a false
    // result is all bits clear. For a 1-bit element type these are the same
    // bit pattern as 1 and 0, respectively.
    elements[i] =
        cir::IntAttr::get(getType().getElementType(), cmpResult ? -1LL : 0LL);
  }

  return cir::ConstVectorAttr::get(
      getType(), mlir::ArrayAttr::get(getContext(), elements));
}

//===----------------------------------------------------------------------===//
// VecShuffleOp
//===----------------------------------------------------------------------===//

OpFoldResult cir::VecShuffleOp::fold(FoldAdaptor adaptor) {
  auto vec1Attr =
      mlir::dyn_cast_if_present<cir::ConstVectorAttr>(adaptor.getVec1());
  auto vec2Attr =
      mlir::dyn_cast_if_present<cir::ConstVectorAttr>(adaptor.getVec2());
  if (!vec1Attr || !vec2Attr)
    return {};

  mlir::Type vec1ElemTy =
      mlir::cast<cir::VectorType>(vec1Attr.getType()).getElementType();

  mlir::ArrayAttr vec1Elts = vec1Attr.getElts();
  mlir::ArrayAttr vec2Elts = vec2Attr.getElts();
  mlir::ArrayAttr indicesElts = adaptor.getIndices();

  SmallVector<mlir::Attribute, 16> elements;
  elements.reserve(indicesElts.size());

  uint64_t vec1Size = vec1Elts.size();
  for (const auto &idxAttr : indicesElts.getAsRange<cir::IntAttr>()) {
    if (idxAttr.getSInt() == -1) {
      elements.push_back(cir::UndefAttr::get(vec1ElemTy));
      continue;
    }

    uint64_t idxValue = idxAttr.getUInt();
    elements.push_back(idxValue < vec1Size ? vec1Elts[idxValue]
                                           : vec2Elts[idxValue - vec1Size]);
  }

  return cir::ConstVectorAttr::get(
      getType(), mlir::ArrayAttr::get(getContext(), elements));
}

LogicalResult cir::VecShuffleOp::verify() {
  if (getVec1().getType().isScalable() || getVec2().getType().isScalable() ||
      getResult().getType().isScalable())
    return emitOpError() << "scalable vectors are not supported";

  // The number of elements in the indices array must match the number of
  // elements in the result type.
  if (getIndices().size() != getResult().getType().getFixedSize()) {
    return emitOpError() << ": the number of elements in " << getIndices()
                         << " and " << getResult().getType() << " don't match";
  }

  // The element types of the two input vectors and of the result type must
  // match.
  if (getVec1().getType().getElementType() !=
      getResult().getType().getElementType()) {
    return emitOpError() << ": element types of " << getVec1().getType()
                         << " and " << getResult().getType() << " don't match";
  }

  const uint64_t maxValidIndex = getVec1().getType().getFixedSize() +
                                 getVec2().getType().getFixedSize() - 1;
  if (llvm::any_of(
          getIndices().getAsRange<cir::IntAttr>(), [&](cir::IntAttr idxAttr) {
            return idxAttr.getSInt() != -1 && idxAttr.getUInt() > maxValidIndex;
          })) {
    return emitOpError() << ": index for __builtin_shufflevector must be "
                            "less than the total number of vector elements";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// VecShuffleDynamicOp
//===----------------------------------------------------------------------===//

OpFoldResult cir::VecShuffleDynamicOp::fold(FoldAdaptor adaptor) {
  mlir::Attribute vec = adaptor.getVec();
  mlir::Attribute indices = adaptor.getIndices();
  if (mlir::isa_and_nonnull<cir::ConstVectorAttr>(vec) &&
      mlir::isa_and_nonnull<cir::ConstVectorAttr>(indices)) {
    auto vecAttr = mlir::cast<cir::ConstVectorAttr>(vec);
    auto indicesAttr = mlir::cast<cir::ConstVectorAttr>(indices);

    mlir::ArrayAttr vecElts = vecAttr.getElts();
    mlir::ArrayAttr indicesElts = indicesAttr.getElts();

    const uint64_t numElements = vecElts.size();

    SmallVector<mlir::Attribute, 16> elements;
    elements.reserve(numElements);

    const uint64_t maskBits = llvm::NextPowerOf2(numElements - 1) - 1;
    for (const auto &idxAttr : indicesElts.getAsRange<cir::IntAttr>()) {
      uint64_t idxValue = idxAttr.getUInt();
      uint64_t newIdx = idxValue & maskBits;
      elements.push_back(vecElts[newIdx]);
    }

    return cir::ConstVectorAttr::get(
        getType(), mlir::ArrayAttr::get(getContext(), elements));
  }

  return {};
}

LogicalResult cir::VecShuffleDynamicOp::verify() {
  // The number of elements in the two input vectors must match.
  if (getVec().getType().getSize() !=
      mlir::cast<cir::VectorType>(getIndices().getType()).getSize()) {
    return emitOpError() << ": the number of elements in " << getVec().getType()
                         << " and " << getIndices().getType() << " don't match";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// VecTernaryOp
//===----------------------------------------------------------------------===//

LogicalResult cir::VecTernaryOp::verify() {
  // Verify that the condition operand has the same number of elements as the
  // other operands.  (The automatic verification already checked that all
  // operands are vector types and that the second and third operands are the
  // same type.)
  if (getCond().getType().getSize() != getLhs().getType().getSize()) {
    return emitOpError() << ": the number of elements in "
                         << getCond().getType() << " and " << getLhs().getType()
                         << " don't match";
  }
  return success();
}

OpFoldResult cir::VecTernaryOp::fold(FoldAdaptor adaptor) {
  mlir::Attribute cond = adaptor.getCond();
  mlir::Attribute lhs = adaptor.getLhs();
  mlir::Attribute rhs = adaptor.getRhs();

  if (!mlir::isa_and_nonnull<cir::ConstVectorAttr>(cond) ||
      !mlir::isa_and_nonnull<cir::ConstVectorAttr>(lhs) ||
      !mlir::isa_and_nonnull<cir::ConstVectorAttr>(rhs))
    return {};
  auto condVec = mlir::cast<cir::ConstVectorAttr>(cond);
  auto lhsVec = mlir::cast<cir::ConstVectorAttr>(lhs);
  auto rhsVec = mlir::cast<cir::ConstVectorAttr>(rhs);

  mlir::ArrayAttr condElts = condVec.getElts();

  SmallVector<mlir::Attribute, 16> elements;
  elements.reserve(condElts.size());

  for (const auto &[idx, condAttr] :
       llvm::enumerate(condElts.getAsRange<cir::IntAttr>())) {
    if (condAttr.getSInt()) {
      elements.push_back(lhsVec.getElts()[idx]);
    } else {
      elements.push_back(rhsVec.getElts()[idx]);
    }
  }

  cir::VectorType vecTy = getLhs().getType();
  return cir::ConstVectorAttr::get(
      vecTy, mlir::ArrayAttr::get(getContext(), elements));
}

//===----------------------------------------------------------------------===//
// MatrixTransposeOp
//===----------------------------------------------------------------------===//

LogicalResult cir::MatrixTransposeOp::verify() {
  cir::MatrixType valueTy = getValue().getType();
  cir::MatrixType resultTy = getResult().getType();

  if ((valueTy.getElementType() != resultTy.getElementType()) ||
      (valueTy.getNumRows() != resultTy.getNumColumns()) ||
      (valueTy.getNumColumns() != resultTy.getNumRows())) {
    auto expectedTy =
        cir::MatrixType::get(valueTy.getElementType(), valueTy.getNumColumns(),
                             valueTy.getNumRows());
    emitOpError() << "operand type " << valueTy << " expects result type of "
                  << expectedTy << " but got " << resultTy;
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen-generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsVectorMatrix.cpp.inc"
