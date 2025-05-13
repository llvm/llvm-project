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

namespace cir::acc {

template <>
mlir::acc::VariableTypeCategory
OpenACCPointerLikeModel<cir::PointerType>::getPointeeTypeCategory(
    mlir::Type pointer, mlir::TypedValue<mlir::acc::PointerLikeType> varPtr,
    mlir::Type varType) const {
  mlir::Type eleTy = mlir::cast<cir::PointerType>(pointer).getPointee();

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
