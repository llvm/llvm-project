//===-- CIRGenFunction.h - Per-Function state for CIR gen -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the internal per-function state used for CIR translation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRGENFUNCTION_H
#define LLVM_CLANG_LIB_CIR_CIRGENFUNCTION_H

#include "CIRGenValue.h"
#include "mlir/IR/Value.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"

namespace clang {
class Expr;
} // namespace clang

namespace cir {
class CIRGenModule;

// FIXME: for now we are reusing this from lib/Clang/CodeGenFunction.h, which
// isn't available in the include dir. Same for getEvaluationKind below.
enum TypeEvaluationKind { TEK_Scalar, TEK_Complex, TEK_Aggregate };

class CIRGenFunction {
public:
  /// If a return statement is being visited, this holds the return statment's
  /// result expression.
  const clang::Expr *RetExpr = nullptr;

  mlir::Value RetValue = nullptr;
  std::optional<mlir::Location> RetLoc;

  mlir::Type FnRetTy;
  clang::QualType FnRetQualTy;

  CIRGenModule &CGM;
  clang::ASTContext &getContext() const;

  /// Sanitizers enabled for this function.
  clang::SanitizerSet SanOpts;

  ///  Return the TypeEvaluationKind of QualType \c T.
  static TypeEvaluationKind getEvaluationKind(clang::QualType T);

  static bool hasScalarEvaluationKind(clang::QualType T) {
    return getEvaluationKind(T) == TEK_Scalar;
  }

  static bool hasAggregateEvaluationKind(clang::QualType T) {
    return getEvaluationKind(T) == TEK_Aggregate;
  }

  CIRGenFunction(CIRGenModule &CGM);

  // TODO: This is currently just a dumb stub. But we want to be able to clearly
  // assert where we arne't doing things that we know we should and will crash
  // as soon as we add a DebugInfo type to this class.
  std::nullptr_t *getDebugInfo() { return nullptr; }

  /// buildAnyExpr - Emit code to compute the specified expression which can
  /// have any type. The result is returned as an RValue struct. If this is an
  /// aggregate expression, the aggloc/agglocvolatile arguments indicate where
  /// the result should be returned.
  /// TODO: if this is an aggregate expression, add a AggValueSlot to indicate
  /// where the result should be returned.
  RValue buildAnyExpr(const clang::Expr *E);
};

} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_CIRGENFUNCTION_H
