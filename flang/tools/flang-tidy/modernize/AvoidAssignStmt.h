//===--- AvoidAssignStmt.h - flang-tidy -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_TOOLS_FLANG_TIDY_MODERNIZE_AVOIDASSIGNSTMT_H
#define LLVM_FLANG_TOOLS_FLANG_TIDY_MODERNIZE_AVOIDASSIGNSTMT_H

#include "../FlangTidyCheck.h"
#include "flang/Parser/parse-tree.h"

namespace Fortran::tidy::modernize {

/// This check verifies that ASSIGN and ASSIGNED GOTO statements are avoided.
///
/// For the user-facing documentation see:
/// https://flang.llvm.org/@PLACEHOLDER@/avoid-assign-stmt.html
class AvoidAssignStmtCheck : public virtual FlangTidyCheck {
public:
  using FlangTidyCheck::FlangTidyCheck;
  virtual ~AvoidAssignStmtCheck() = default;
  void Enter(const parser::AssignStmt &) override;
  void Enter(const parser::AssignedGotoStmt &) override;
};

} // namespace Fortran::tidy::modernize

#endif // LLVM_FLANG_TOOLS_FLANG_TIDY_MODERNIZE_AVOIDASSIGNSTMT_H
