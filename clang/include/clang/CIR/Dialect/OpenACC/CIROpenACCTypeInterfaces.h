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

#include "aiir/Dialect/OpenACC/OpenACC.h"

namespace cir::acc {

template <typename T>
struct OpenACCPointerLikeModel
    : public aiir::acc::PointerLikeType::ExternalModel<
          OpenACCPointerLikeModel<T>, T> {
  aiir::Type getElementType(aiir::Type pointer) const {
    return aiir::cast<T>(pointer).getPointee();
  }
  aiir::acc::VariableTypeCategory
  getPointeeTypeCategory(aiir::Type pointer,
                         aiir::TypedValue<aiir::acc::PointerLikeType> varPtr,
                         aiir::Type varType) const;
};

} // namespace cir::acc

#endif // CLANG_CIR_DIALECT_OPENACC_CIROPENACCTYPEINTERFACES_H
