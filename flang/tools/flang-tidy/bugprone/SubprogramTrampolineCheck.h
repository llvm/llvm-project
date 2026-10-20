//===--- SubprogramTrampolineCheck.h - flang-tidy ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_TOOLS_FLANG_TIDY_BUGPRONE_SUBPROGRAMTRAMPOLINECHECK_H
#define LLVM_FLANG_TOOLS_FLANG_TIDY_BUGPRONE_SUBPROGRAMTRAMPOLINECHECK_H

#include "../FlangTidyCheck.h"
#include "flang/Parser/parse-tree.h"

namespace Fortran::tidy::bugprone {

/// This check verifies that a contained subprogram is not passed as an
/// actual argument to a procedure.
///
/// For the user-facing documentation see:
/// https://flang.llvm.org/@PLACEHOLDER@/subprogram-trampoline.html
class SubprogramTrampolineCheck : public virtual FlangTidyCheck {
public:
  using FlangTidyCheck::FlangTidyCheck;
  virtual ~SubprogramTrampolineCheck() = default;
  void Enter(const parser::Expr &) override;
  void Enter(const parser::CallStmt &) override;
};

} // namespace Fortran::tidy::bugprone

#endif // LLVM_FLANG_TOOLS_FLANG_TIDY_BUGPRONE_SUBPROGRAMTRAMPOLINECHECK_H
