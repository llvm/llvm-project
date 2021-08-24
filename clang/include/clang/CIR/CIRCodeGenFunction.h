//===-- CIRCIRCodeGenFunction.h - Per-Function state for LLVM CodeGen -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the internal per-function state used for cir translation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRCODEGENFUNCTION_H
#define LLVM_CLANG_LIB_CIR_CIRCODEGENFUNCTION_H

#include "mlir/IR/Value.h"
#include "clang/AST/Type.h"

namespace clang {
class Expr;
}

using namespace clang;

namespace cir {

// FIXME: for now we are reusing this from lib/Clang/CodeGenFunction.h, which
// isn't available in the include dir. Same for getEvaluationKind below.
enum TypeEvaluationKind { TEK_Scalar, TEK_Complex, TEK_Aggregate };

/// The source of the alignment of an l-value; an expression of
/// confidence in the alignment actually matching the estimate.
enum class AlignmentSource {
  /// The l-value was an access to a declared entity or something
  /// equivalently strong, like the address of an array allocated by a
  /// language runtime.
  Decl,

  /// The l-value was considered opaque, so the alignment was
  /// determined from a type, but that type was an explicitly-aligned
  /// typedef.
  AttributedType,

  /// The l-value was considered opaque, so the alignment was
  /// determined from a type.
  Type
};

/// Given that the base address has the given alignment source, what's
/// our confidence in the alignment of the field?
static inline AlignmentSource getFieldAlignmentSource(AlignmentSource Source) {
  // For now, we don't distinguish fields of opaque pointers from
  // top-level declarations, but maybe we should.
  return AlignmentSource::Decl;
}

class LValueBaseInfo {
  AlignmentSource AlignSource;

public:
  explicit LValueBaseInfo(AlignmentSource Source = AlignmentSource::Type)
      : AlignSource(Source) {}
  AlignmentSource getAlignmentSource() const { return AlignSource; }
  void setAlignmentSource(AlignmentSource Source) { AlignSource = Source; }

  void mergeForCast(const LValueBaseInfo &Info) {
    setAlignmentSource(Info.getAlignmentSource());
  }
};

class CIRCodeGenFunction {
public:
  /// If a return statement is being visited, this holds the return statment's
  /// result expression.
  const Expr *RetExpr = nullptr;

  mlir::Value RetValue = nullptr;
  mlir::Type FnRetTy;
  clang::QualType FnRetQualTy;

  ///  Return the TypeEvaluationKind of QualType \c T.
  static TypeEvaluationKind getEvaluationKind(QualType T);

  static bool hasScalarEvaluationKind(QualType T) {
    return getEvaluationKind(T) == TEK_Scalar;
  }

  static bool hasAggregateEvaluationKind(QualType T) {
    return getEvaluationKind(T) == TEK_Aggregate;
  }

  CIRCodeGenFunction();
};

} // namespace cir

#endif