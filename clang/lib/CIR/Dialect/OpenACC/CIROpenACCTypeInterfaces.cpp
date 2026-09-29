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

mlir::Type getBaseType(mlir::Value varPtr) {
  mlir::Operation *op = varPtr.getDefiningOp();
  assert(op && "Expected a defining operation");

  // This is the variable definition we're looking for.
  if (auto allocaOp = mlir::dyn_cast<cir::AllocaOp>(*op))
    return allocaOp.getAllocaType();

  // Look through casts to the source pointer.
  if (auto castOp = mlir::dyn_cast<cir::CastOp>(*op))
    return getBaseType(castOp.getSrc());

  // Follow the source of ptr strides.
  if (auto ptrStrideOp = mlir::dyn_cast<cir::PtrStrideOp>(*op))
    return getBaseType(ptrStrideOp.getBase());

  if (auto getMemberOp = mlir::dyn_cast<cir::GetMemberOp>(*op))
    return getBaseType(getMemberOp.getAddr());

  return mlir::cast<cir::PointerType>(varPtr.getType()).getPointee();
}

template <>
mlir::acc::VariableTypeCategory
OpenACCPointerLikeModel<cir::PointerType>::getPointeeTypeCategory(
    mlir::Type pointer, mlir::TypedValue<mlir::acc::PointerLikeType> varPtr,
    mlir::Type varType) const {
  mlir::Type eleTy = getBaseType(varPtr);

  if (auto mappableTy = mlir::dyn_cast<mlir::acc::MappableType>(eleTy))
    return mappableTy.getTypeCategory(varPtr);

  if (isAnyIntegerOrFloatingPointType(eleTy) ||
      mlir::isa<cir::BoolType>(eleTy) || mlir::isa<cir::PointerType>(eleTy))
    return mlir::acc::VariableTypeCategory::scalar;
  if (mlir::isa<cir::ArrayType>(eleTy))
    return mlir::acc::VariableTypeCategory::array;
  if (mlir::isa<cir::RecordType>(eleTy))
    return mlir::acc::VariableTypeCategory::composite;
  if (mlir::isa<cir::FuncType>(eleTy) || mlir::isa<cir::VectorType>(eleTy))
    return mlir::acc::VariableTypeCategory::nonscalar;

  // Without further checking, this type cannot be categorized.
  return mlir::acc::VariableTypeCategory::uncategorized;
}

} // namespace cir::acc
