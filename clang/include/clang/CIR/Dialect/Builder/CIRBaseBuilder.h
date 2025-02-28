//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_BUILDER_CIRBASEBUILDER_H
#define LLVM_CLANG_CIR_DIALECT_BUILDER_CIRBASEBUILDER_H

#include "clang/AST/CharUnits.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace cir {

class CIRBaseBuilderTy : public mlir::OpBuilder {

public:
  CIRBaseBuilderTy(mlir::MLIRContext &mlirContext)
      : mlir::OpBuilder(&mlirContext) {}

  cir::ConstantOp getBool(bool state, mlir::Location loc) {
    return create<cir::ConstantOp>(loc, getBoolTy(), getCIRBoolAttr(state));
  }
  cir::ConstantOp getFalse(mlir::Location loc) { return getBool(false, loc); }
  cir::ConstantOp getTrue(mlir::Location loc) { return getBool(true, loc); }

  cir::BoolType getBoolTy() { return cir::BoolType::get(getContext()); }

  cir::PointerType getPointerTo(mlir::Type ty) {
    return cir::PointerType::get(getContext(), ty);
  }

  cir::PointerType getVoidPtrTy() {
    return getPointerTo(cir::VoidType::get(getContext()));
  }

  cir::BoolAttr getCIRBoolAttr(bool state) {
    return cir::BoolAttr::get(getContext(), getBoolTy(), state);
  }

  mlir::TypedAttr getConstPtrAttr(mlir::Type type, int64_t value) {
    auto valueAttr = mlir::IntegerAttr::get(
        mlir::IntegerType::get(type.getContext(), 64), value);
    return cir::ConstPtrAttr::get(
        getContext(), mlir::cast<cir::PointerType>(type), valueAttr);
  }

  mlir::Value createAlloca(mlir::Location loc, cir::PointerType addrType,
                           mlir::Type type, llvm::StringRef name,
                           mlir::IntegerAttr alignment) {
    return create<cir::AllocaOp>(loc, addrType, type, name, alignment);
  }

  cir::LoadOp createLoad(mlir::Location loc, mlir::Value ptr,
                         bool isVolatile = false, uint64_t alignment = 0) {
    mlir::IntegerAttr intAttr;
    if (alignment)
      intAttr = mlir::IntegerAttr::get(
          mlir::IntegerType::get(ptr.getContext(), 64), alignment);

    return create<cir::LoadOp>(loc, ptr);
  }

  //
  // Block handling helpers
  // ----------------------
  //
  static OpBuilder::InsertPoint getBestAllocaInsertPoint(mlir::Block *block) {
    auto last =
        std::find_if(block->rbegin(), block->rend(), [](mlir::Operation &op) {
          // TODO: Add LabelOp missing feature here
          return mlir::isa<cir::AllocaOp>(&op);
        });

    if (last != block->rend())
      return OpBuilder::InsertPoint(block, ++mlir::Block::iterator(&*last));
    return OpBuilder::InsertPoint(block, block->begin());
  };

  mlir::IntegerAttr getSizeFromCharUnits(mlir::MLIRContext *ctx,
                                         clang::CharUnits size) {
    // Note that mlir::IntegerType is used instead of cir::IntType here
    // because we don't need sign information for this to be useful, so keep
    // it simple.
    return mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                  size.getQuantity());
  }
};

} // namespace cir

#endif
