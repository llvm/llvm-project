//===--- StmtUtils.cpp - Shared statement emission queries ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGenUtils/StmtUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"

namespace clang::CodeGenUtils {

bool checkIfLoopMustProgress(const LangOptions &LangOpts,
                             const CodeGenOptions &CGOpts, ASTContext &Ctx,
                             const Expr *ControllingExpression,
                             bool HasEmptyBody,
                             llvm::function_ref<void()> RemoveMustProgress) {
  if (CGOpts.getFiniteLoops() == CodeGenOptions::FiniteLoopsKind::Never)
    return false;

  // Now apply rules for plain C (see  6.8.5.6 in C11).
  // Loops with constant conditions do not have to make progress in any C
  // version.
  // As an extension, we consisider loops whose constant expression
  // can be constant-folded.
  Expr::EvalResult Result;
  bool CondIsConstInt =
      !ControllingExpression ||
      (ControllingExpression->EvaluateAsInt(Result, Ctx) && Result.Val.isInt());

  bool CondIsTrue = CondIsConstInt && (!ControllingExpression ||
                                       Result.Val.getInt().getBoolValue());

  // Loops with non-constant conditions must make progress in C11 and later.
  if (LangOpts.C11 && !CondIsConstInt)
    return true;

  // [C++26][intro.progress] (DR)
  // The implementation may assume that any thread will eventually do one of the
  // following:
  // [...]
  // - continue execution of a trivial infinite loop ([stmt.iter.general]).
  if (CGOpts.getFiniteLoops() == CodeGenOptions::FiniteLoopsKind::Always ||
      LangOpts.CPlusPlus11) {
    if (HasEmptyBody && CondIsTrue) {
      RemoveMustProgress();
      return false;
    }
    return true;
  }
  return false;
}

} // namespace clang::CodeGenUtils
