//===-- Tools/PointerModels.h --------------------- *-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_TOOLS_POINTER_MODELS_H
#define FORTRAN_TOOLS_POINTER_MODELS_H

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

/// models for FIR pointer like types that already provide a `getElementType` or
/// a `getEleTy` method

template <typename T>
struct PointerLikeModel
    : public mlir::omp::PointerLikeType::ExternalModel<PointerLikeModel<T>, T> {
  mlir::Type getElementType(mlir::Type pointer) const {
    return pointer.cast<T>().getElementType();
  }
};

template <typename T>
struct AlternativePointerLikeModel
    : public mlir::omp::PointerLikeType::ExternalModel<
          AlternativePointerLikeModel<T>, T> {
  mlir::Type getElementType(mlir::Type pointer) const {
    return pointer.cast<T>().getEleTy();
  }
};

#endif // FORTRAN_TOOLS_POINTER_MODELS_H
