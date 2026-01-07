//===------ EvaluationResult.h - Result class  for the VM -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_EVALUATION_RESULT_H
#define LLVM_CLANG_AST_INTERP_EVALUATION_RESULT_H

#include "clang/AST/APValue.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"

namespace clang {
namespace interp {
class EvalEmitter;
class Context;
class Pointer;
class SourceInfo;
class InterpState;

/// Defines the result of an evaluation.
///
/// The Kind defined if the evaluation was invalid, valid (but empty, e.g. for
/// void expressions) or if we have a valid evaluation result.
///
/// We use this class to inspect and diagnose the result, as well as
/// convert it to the requested form.
class EvaluationResult final {
public:
  enum ResultKind {
    Empty,   // Initial state.
    Invalid, // Result is invalid.
    Valid,   // Result is valid and empty.
  };

  using DeclTy = llvm::PointerUnion<const Decl *, const Expr *>;

private:
#ifndef NDEBUG
  const Context *Ctx = nullptr;
#endif
  APValue Value;
  ResultKind Kind = Empty;
  DeclTy Source = nullptr;

  void setSource(DeclTy D) { Source = D; }

  void takeValue(APValue &&V) {
    assert(empty());
    Value = std::move(V);
  }
  void setInvalid() {
    // We are NOT asserting empty() here, since setting it to invalid
    // is allowed even if there is already a result.
    Kind = Invalid;
  }
  void setValid() {
    assert(empty());
    Kind = Valid;
  }

public:
#ifndef NDEBUG
  EvaluationResult(const Context *Ctx) : Ctx(Ctx) {}
#else
  EvaluationResult(const Context *Ctx) {}
#endif

  bool empty() const { return Kind == Empty; }
  bool isInvalid() const { return Kind == Invalid; }

  /// Returns an APValue for the evaluation result.
  APValue toAPValue() const {
    assert(!empty());
    assert(!isInvalid());
    return Value;
  }

  APValue stealAPValue() { return std::move(Value); }

  /// Check that all subobjects of the given pointer have been initialized.
  bool checkFullyInitialized(InterpState &S, const Pointer &Ptr) const;
  /// Check that none of the blocks the given pointer (transitively) points
  /// to are dynamically allocated.
  bool checkReturnValue(InterpState &S, const Context &Ctx, const Pointer &Ptr,
                        const SourceInfo &Info);

  QualType getSourceType() const {
    if (const auto *D =
            dyn_cast_if_present<ValueDecl>(Source.dyn_cast<const Decl *>()))
      return D->getType();
    if (const auto *E = Source.dyn_cast<const Expr *>())
      return E->getType();
    return QualType();
  }

  /// Dump to stderr.
  void dump() const;

  friend class EvalEmitter;
  friend class InterpState;
};

} // namespace interp
} // namespace clang

#endif
