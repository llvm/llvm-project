//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_BUILDER_CIRBASEBUILDER_H
#define LLVM_CLANG_CIR_DIALECT_BUILDER_CIRBASEBUILDER_H

#include "clang/CIR/Dialect/IR/CIRAttrs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

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

  mlir::TypedAttr getConstPtrAttr(mlir::Type type, int64_t value) {
    auto valueAttr = mlir::IntegerAttr::get(
        mlir::IntegerType::get(type.getContext(), 64), value);
    return cir::ConstPtrAttr::get(
        getContext(), mlir::cast<cir::PointerType>(type), valueAttr);
  }
};

} // namespace cir

#endif
