//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of external dialect interfaces for CIR.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/OpenACC/CIROpenACCTypeInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

namespace cir::acc {

aiir::Type getBaseType(aiir::Value varPtr) {
  aiir::Operation *op = varPtr.getDefiningOp();
  assert(op && "Expected a defining operation");

  // This is the variable definition we're looking for.
  if (auto allocaOp = aiir::dyn_cast<cir::AllocaOp>(*op))
    return allocaOp.getAllocaType();

  // Look through casts to the source pointer.
  if (auto castOp = aiir::dyn_cast<cir::CastOp>(*op))
    return getBaseType(castOp.getSrc());

  // Follow the source of ptr strides.
  if (auto ptrStrideOp = aiir::dyn_cast<cir::PtrStrideOp>(*op))
    return getBaseType(ptrStrideOp.getBase());

  if (auto getMemberOp = aiir::dyn_cast<cir::GetMemberOp>(*op))
    return getBaseType(getMemberOp.getAddr());

  return aiir::cast<cir::PointerType>(varPtr.getType()).getPointee();
}

template <>
aiir::acc::VariableTypeCategory
OpenACCPointerLikeModel<cir::PointerType>::getPointeeTypeCategory(
    aiir::Type pointer, aiir::TypedValue<aiir::acc::PointerLikeType> varPtr,
    aiir::Type varType) const {
  aiir::Type eleTy = getBaseType(varPtr);

  if (auto mappableTy = aiir::dyn_cast<aiir::acc::MappableType>(eleTy))
    return mappableTy.getTypeCategory(varPtr);

  if (isAnyIntegerOrFloatingPointType(eleTy) ||
      aiir::isa<cir::BoolType>(eleTy) || aiir::isa<cir::PointerType>(eleTy))
    return aiir::acc::VariableTypeCategory::scalar;
  if (aiir::isa<cir::ArrayType>(eleTy))
    return aiir::acc::VariableTypeCategory::array;
  if (aiir::isa<cir::RecordType>(eleTy))
    return aiir::acc::VariableTypeCategory::composite;
  if (aiir::isa<cir::FuncType>(eleTy) || aiir::isa<cir::VectorType>(eleTy))
    return aiir::acc::VariableTypeCategory::nonscalar;

  // Without further checking, this type cannot be categorized.
  return aiir::acc::VariableTypeCategory::uncategorized;
}

} // namespace cir::acc
