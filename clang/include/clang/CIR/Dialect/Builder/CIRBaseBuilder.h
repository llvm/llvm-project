//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_BUILDER_CIRBASEBUILDER_H
#define LLVM_CLANG_CIR_DIALECT_BUILDER_CIRBASEBUILDER_H

#include "mlir/IR/Builders.h"

namespace cir {

class CIRBaseBuilderTy : public mlir::OpBuilder {

public:
  CIRBaseBuilderTy(mlir::MLIRContext &mlirContext)
      : mlir::OpBuilder(&mlirContext) {}

  cir::PointerType getPointerTo(mlir::Type ty) {
    return cir::PointerType::get(getContext(), ty);
  }

  cir::PointerType getVoidPtrTy() {
    return getPointerTo(cir::VoidType::get(getContext()));
  }
};

} // namespace cir

#endif
