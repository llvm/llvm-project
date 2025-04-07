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
#include "clang/AST/Type.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/ErrorHandling.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace cir {

class CIRBaseBuilderTy : public mlir::OpBuilder {

public:
  CIRBaseBuilderTy(mlir::MLIRContext &mlirContext)
      : mlir::OpBuilder(&mlirContext) {}

  cir::ConstantOp getConstant(mlir::Location loc, mlir::TypedAttr attr) {
    return create<cir::ConstantOp>(loc, attr.getType(), attr);
  }

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

  /// Create a for operation.
  cir::ForOp createFor(
      mlir::Location loc,
      llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> condBuilder,
      llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> bodyBuilder,
      llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> stepBuilder) {
    return create<cir::ForOp>(loc, condBuilder, bodyBuilder, stepBuilder);
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

  cir::StoreOp createStore(mlir::Location loc, mlir::Value val,
                           mlir::Value dst) {
    return create<cir::StoreOp>(loc, val, dst);
  }

  mlir::Value createDummyValue(mlir::Location loc, mlir::Type type,
                               clang::CharUnits alignment) {
    auto addr = createAlloca(loc, getPointerTo(type), type, {},
                             getSizeFromCharUnits(getContext(), alignment));
    return createLoad(loc, addr);
  }

  //===--------------------------------------------------------------------===//
  // Cast/Conversion Operators
  //===--------------------------------------------------------------------===//

  mlir::Value createCast(mlir::Location loc, cir::CastKind kind,
                         mlir::Value src, mlir::Type newTy) {
    if (newTy == src.getType())
      return src;
    return create<cir::CastOp>(loc, newTy, kind, src);
  }

  mlir::Value createCast(cir::CastKind kind, mlir::Value src,
                         mlir::Type newTy) {
    if (newTy == src.getType())
      return src;
    return createCast(src.getLoc(), kind, src, newTy);
  }

  mlir::Value createIntCast(mlir::Value src, mlir::Type newTy) {
    return createCast(cir::CastKind::integral, src, newTy);
  }

  mlir::Value createIntToPtr(mlir::Value src, mlir::Type newTy) {
    return createCast(cir::CastKind::int_to_ptr, src, newTy);
  }

  mlir::Value createPtrToInt(mlir::Value src, mlir::Type newTy) {
    return createCast(cir::CastKind::ptr_to_int, src, newTy);
  }

  mlir::Value createPtrToBoolCast(mlir::Value v) {
    return createCast(cir::CastKind::ptr_to_bool, v, getBoolTy());
  }

  mlir::Value createBoolToInt(mlir::Value src, mlir::Type newTy) {
    return createCast(cir::CastKind::bool_to_int, src, newTy);
  }

  mlir::Value createBitcast(mlir::Value src, mlir::Type newTy) {
    return createCast(cir::CastKind::bitcast, src, newTy);
  }

  mlir::Value createBitcast(mlir::Location loc, mlir::Value src,
                            mlir::Type newTy) {
    return createCast(loc, cir::CastKind::bitcast, src, newTy);
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

  /// Create a loop condition.
  cir::ConditionOp createCondition(mlir::Value condition) {
    return create<cir::ConditionOp>(condition.getLoc(), condition);
  }

  /// Create a yield operation.
  cir::YieldOp createYield(mlir::Location loc, mlir::ValueRange value = {}) {
    return create<cir::YieldOp>(loc, value);
  }
};

} // namespace cir

#endif
