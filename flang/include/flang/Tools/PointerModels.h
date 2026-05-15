//===-- Tools/PointerModels.h --------------------- *-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_TOOLS_POINTER_MODELS_H
#define FORTRAN_TOOLS_POINTER_MODELS_H

#include "mlir/IR/BuiltinTypeInterfaces.h"

/// Models for FIR pointer like types that already provide a `getElementType`
/// method. These implement the core MLIR PtrLikeTypeInterface.

template <typename T>
struct OpenMPPointerLikeModel
    : public mlir::PtrLikeTypeInterface::ExternalModel<
          OpenMPPointerLikeModel<T>, T> {
  mlir::Attribute getMemorySpace(mlir::Type pointer) const {
    return mlir::Attribute();
  }
  mlir::Type getElementType(mlir::Type pointer) const {
    return mlir::cast<T>(pointer).getElementType();
  }
  bool hasPtrMetadata(mlir::Type pointer) const { return false; }
  mlir::FailureOr<mlir::PtrLikeTypeInterface> clonePtrWith(mlir::Type pointer,
      mlir::Attribute memorySpace,
      std::optional<mlir::Type> elementType) const {
    return mlir::failure();
  }
};

#endif // FORTRAN_TOOLS_POINTER_MODELS_H
