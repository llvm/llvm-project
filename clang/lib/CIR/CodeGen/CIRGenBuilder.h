//===-- CIRGenBuilder.h - CIRBuilder implementation  ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRGENBUILDER_H
#define LLVM_CLANG_LIB_CIR_CIRGENBUILDER_H

#include "clang/CIR/Dialect/IR/FPEnv.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/FloatingPointMode.h"

namespace cir {

class CIRGenFunction;

class CIRGenBuilderTy : public mlir::OpBuilder {
  fp::ExceptionBehavior DefaultConstrainedExcept = fp::ebStrict;
  llvm::RoundingMode DefaultConstrainedRounding = llvm::RoundingMode::Dynamic;

public:
  CIRGenBuilderTy(mlir::MLIRContext &C) : mlir::OpBuilder(&C) {}

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
};

} // namespace cir

#endif
