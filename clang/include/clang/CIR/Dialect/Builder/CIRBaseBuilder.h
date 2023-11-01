//===-- CIRBaseBuilder.h - CIRBuilder implementation  -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIRBASEBUILDER_H
#define LLVM_CLANG_LIB_CIRBASEBUILDER_H

#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/IR/FPEnv.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <optional>
#include <string>

namespace cir {

class CIRBaseBuilderTy : public mlir::OpBuilder {

public:
  CIRBaseBuilderTy(mlir::MLIRContext &C) : mlir::OpBuilder(&C) {}

  mlir::Value getConstAPInt(mlir::Location loc, mlir::Type typ,
                            const llvm::APInt &val) {
    return create<mlir::cir::ConstantOp>(loc, typ,
                                         getAttr<mlir::cir::IntAttr>(typ, val));
  }

  mlir::Value createNot(mlir::Value value) {
    return create<mlir::cir::UnaryOp>(value.getLoc(), value.getType(),
                                      mlir::cir::UnaryOpKind::Not, value);
  }

  mlir::Value createBinop(mlir::Value lhs, mlir::cir::BinOpKind kind,
                          const llvm::APInt &rhs) {
    return create<mlir::cir::BinOp>(
        lhs.getLoc(), lhs.getType(), kind, lhs,
        getConstAPInt(lhs.getLoc(), lhs.getType(), rhs));
  }

  mlir::Value createBinop(mlir::Value lhs, mlir::cir::BinOpKind kind,
                          mlir::Value rhs) {
    return create<mlir::cir::BinOp>(lhs.getLoc(), lhs.getType(), kind, lhs,
                                    rhs);
  }

  mlir::Value createShift(mlir::Value lhs, const llvm::APInt &rhs,
                          bool isShiftLeft) {
    return create<mlir::cir::ShiftOp>(
        lhs.getLoc(), lhs.getType(), lhs,
        getConstAPInt(lhs.getLoc(), lhs.getType(), rhs), isShiftLeft);
  }

  mlir::Value createShift(mlir::Value lhs, unsigned bits, bool isShiftLeft) {
    auto width = lhs.getType().dyn_cast<mlir::cir::IntType>().getWidth();
    auto shift = llvm::APInt(width, bits);
    return createShift(lhs, shift, isShiftLeft);
  }

  mlir::Value createShiftLeft(mlir::Value lhs, unsigned bits) {
    return createShift(lhs, bits, true);
  }

  mlir::Value createShiftRight(mlir::Value lhs, unsigned bits) {
    return createShift(lhs, bits, false);
  }

  mlir::Value createLowBitsSet(mlir::Location loc, unsigned size,
                               unsigned bits) {
    auto val = llvm::APInt::getLowBitsSet(size, bits);
    auto typ = mlir::cir::IntType::get(getContext(), size, false);
    return getConstAPInt(loc, typ, val);
  }

  mlir::Value createAnd(mlir::Value lhs, llvm::APInt rhs) {
    auto val = getConstAPInt(lhs.getLoc(), lhs.getType(), rhs);
    return createBinop(lhs, mlir::cir::BinOpKind::And, val);
  }

  mlir::Value createAnd(mlir::Value lhs, mlir::Value rhs) {
    return createBinop(lhs, mlir::cir::BinOpKind::And, rhs);
  }

  mlir::Value createOr(mlir::Value lhs, llvm::APInt rhs) {
    auto val = getConstAPInt(lhs.getLoc(), lhs.getType(), rhs);
    return createBinop(lhs, mlir::cir::BinOpKind::Or, val);
  }

  mlir::Value createOr(mlir::Value lhs, mlir::Value rhs) {
    return createBinop(lhs, mlir::cir::BinOpKind::Or, rhs);
  }

  //===--------------------------------------------------------------------===//
  // Cast/Conversion Operators
  //===--------------------------------------------------------------------===//

  mlir::Value createCast(mlir::cir::CastKind kind, mlir::Value src,
                         mlir::Type newTy) {
    if (newTy == src.getType())
      return src;
    return create<mlir::cir::CastOp>(src.getLoc(), newTy, kind, src);
  }

  mlir::Value createIntCast(mlir::Value src, mlir::Type newTy) {
    return create<mlir::cir::CastOp>(src.getLoc(), newTy,
                                     mlir::cir::CastKind::integral, src);
  }

  mlir::Value createIntToPtr(mlir::Value src, mlir::Type newTy) {
    return create<mlir::cir::CastOp>(src.getLoc(), newTy,
                                     mlir::cir::CastKind::int_to_ptr, src);
  }

  mlir::Value createPtrToInt(mlir::Value src, mlir::Type newTy) {
    return create<mlir::cir::CastOp>(src.getLoc(), newTy,
                                     mlir::cir::CastKind::ptr_to_int, src);
  }

  // TODO(cir): the following function was introduced to keep in sync with LLVM
  // codegen. CIR does not have "zext" operations. It should eventually be
  // renamed or removed. For now, we just add whatever cast is required here.
  mlir::Value createZExtOrBitCast(mlir::Location loc, mlir::Value src,
                                  mlir::Type newTy) {
    auto srcTy = src.getType();

    if (srcTy == newTy)
      return src;

    if (srcTy.isa<mlir::cir::BoolType>() && newTy.isa<mlir::cir::IntType>())
      return createBoolToInt(src, newTy);

    llvm_unreachable("unhandled extension cast");
  }

  mlir::Value createBoolToInt(mlir::Value src, mlir::Type newTy) {
    return createCast(mlir::cir::CastKind::bool_to_int, src, newTy);
  }

  mlir::Value createBitcast(mlir::Value src, mlir::Type newTy) {
    return createCast(mlir::cir::CastKind::bitcast, src, newTy);
  }

  mlir::Value createBitcast(mlir::Location loc, mlir::Value src,
                            mlir::Type newTy) {
    if (newTy == src.getType())
      return src;
    return create<mlir::cir::CastOp>(loc, newTy, mlir::cir::CastKind::bitcast,
                                     src);
  }
};

} // namespace cir
#endif
