//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains external dialect interfaces for CIR.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_OPENACC_CIROPENACCTYPEINTERFACES_H
#define CLANG_CIR_DIALECT_OPENACC_CIROPENACCTYPEINTERFACES_H

#include "mlir/Dialect/OpenACC/OpenACC.h"

namespace cir::acc {

template <typename T>
struct OpenACCPointerLikeModel
    : public mlir::acc::PointerLikeType::ExternalModel<
          OpenACCPointerLikeModel<T>, T> {
  mlir::Type getElementType(mlir::Type pointer) const {
    return mlir::cast<T>(pointer).getPointee();
  }
  mlir::acc::VariableTypeCategory
  getPointeeTypeCategory(mlir::Type pointer,
                         mlir::TypedValue<mlir::acc::PointerLikeType> varPtr,
                         mlir::Type varType) const;
};

} // namespace cir::acc

#endif // CLANG_CIR_DIALECT_OPENACC_CIROPENACCTYPEINTERFACES_H
