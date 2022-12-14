//===-- CIRGenBuilder.h - CIRBuilder implementation  ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRGENBUILDER_H
#define LLVM_CLANG_LIB_CIR_CIRGENBUILDER_H

#include "Address.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/FPEnv.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/FloatingPointMode.h"

namespace cir {

class CIRGenFunction;

class CIRGenBuilderTy : public mlir::OpBuilder {
  bool IsFPConstrained = false;
  fp::ExceptionBehavior DefaultConstrainedExcept = fp::ebStrict;
  llvm::RoundingMode DefaultConstrainedRounding = llvm::RoundingMode::Dynamic;

public:
  CIRGenBuilderTy(mlir::MLIRContext &C) : mlir::OpBuilder(&C) {}

  //
  // Floating point specific helpers
  // -------------------------------
  //

  /// Enable/Disable use of constrained floating point math. When enabled the
  /// CreateF<op>() calls instead create constrained floating point intrinsic
  /// calls. Fast math flags are unaffected by this setting.
  void setIsFPConstrained(bool IsCon) {
    if (IsCon)
      llvm_unreachable("Constrained FP NYI");
    IsFPConstrained = IsCon;
  }

  /// Query for the use of constrained floating point math
  bool getIsFPConstrained() {
    if (IsFPConstrained)
      llvm_unreachable("Constrained FP NYI");
    return IsFPConstrained;
  }

  /// Set the exception handling to be used with constrained floating point
  void setDefaultConstrainedExcept(fp::ExceptionBehavior NewExcept) {
#ifndef NDEBUG
    std::optional<llvm::StringRef> ExceptStr =
        convertExceptionBehaviorToStr(NewExcept);
    assert(ExceptStr && "Garbage strict exception behavior!");
#endif
    DefaultConstrainedExcept = NewExcept;
  }

  /// Set the rounding mode handling to be used with constrained floating point
  void setDefaultConstrainedRounding(llvm::RoundingMode NewRounding) {
#ifndef NDEBUG
    std::optional<llvm::StringRef> RoundingStr =
        convertRoundingModeToStr(NewRounding);
    assert(RoundingStr && "Garbage strict rounding mode!");
#endif
    DefaultConstrainedRounding = NewRounding;
  }

  /// Get the exception handling used with constrained floating point
  fp::ExceptionBehavior getDefaultConstrainedExcept() {
    return DefaultConstrainedExcept;
  }

  /// Get the rounding mode handling used with constrained floating point
  llvm::RoundingMode getDefaultConstrainedRounding() {
    return DefaultConstrainedRounding;
  }

  //
  // Type helpers
  // ------------
  //

  // Fetch the type representing a pointer to an 8-bit integer value.
  mlir::cir::PointerType getInt8PtrTy(unsigned AddrSpace = 0) {
    return mlir::cir::PointerType::get(getContext(),
                                       mlir::IntegerType::get(getContext(), 8));
  }

  /// Get a constant 32-bit value.
  mlir::cir::ConstantOp getInt32(uint32_t C, mlir::Location loc) {
    auto int32Ty = mlir::IntegerType::get(getContext(), 32);
    return create<mlir::cir::ConstantOp>(loc, int32Ty,
                                         mlir::IntegerAttr::get(int32Ty, C));
  }

  //
  // Operation creation helpers
  // --------------------------
  //

  mlir::Value createFPExt(mlir::Value v, mlir::Type destType) {
    if (getIsFPConstrained())
      llvm_unreachable("constrainedfp NYI");

    return create<mlir::cir::CastOp>(v.getLoc(), destType,
                                     mlir::cir::CastKind::floating, v);
  }

  cir::Address createElementBitCast(mlir::Location loc, cir::Address addr,
                                    mlir::Type destType) {
    if (destType == addr.getElementType())
      return addr;

    auto newPtrType = mlir::cir::PointerType::get(getContext(), destType);
    auto cast = create<mlir::cir::CastOp>(
        loc, newPtrType, mlir::cir::CastKind::bitcast, addr.getPointer());
    return Address(cast, addr.getElementType(), addr.getAlignment());
  }

  mlir::Value createLoad(mlir::Location loc, Address addr) {
    return create<mlir::cir::LoadOp>(loc, addr.getElementType(),
                                     addr.getPointer());
  }
};

} // namespace cir

#endif
