//===--------------------------- EvalSettings.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_EVALSETTINGS_H
#define LLVM_CLANG_AST_INTERP_EVALSETTINGS_H

#include "State.h"

namespace clang {
namespace interp {

struct EvalSettings {
  Expr::EvalStatus &EvalStatus;
  const EvaluationMode EvalMode;
  const ConstantExprKind ConstexprKind;

  bool InConstantContext = false;
  bool CheckingPotentialConstantExpression = false;
  bool CheckingForUndefinedBehavior = false;

  EvalSettings(EvaluationMode EvalMode, Expr::EvalStatus &EvalStatus,
               ConstantExprKind ConstexprKind = ConstantExprKind::Normal)
      : EvalStatus(EvalStatus), EvalMode(EvalMode),
        ConstexprKind(ConstexprKind) {}
};

} // namespace interp
} // namespace clang

#endif
