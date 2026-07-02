//===--- MissingActionCheck.h - flang-tidy ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_TOOLS_FLANG_TIDY_BUGPRONE_MISSINGACTIONCHECK_H
#define LLVM_FLANG_TOOLS_FLANG_TIDY_BUGPRONE_MISSINGACTIONCHECK_H

#include "../FlangTidyCheck.h"
#include "flang/Parser/parse-tree.h"

namespace Fortran::tidy::bugprone {

/// This check verifies that all OPEN statements have an ACTION clause, and that
/// the FILE UNIT NUMBER is not a constant literal.
///
/// For the user-facing documentation see:
/// https://flang.llvm.org/@PLACEHOLDER@/missing-action.html
class MissingActionCheck : public virtual FlangTidyCheck {
public:
  using FlangTidyCheck::FlangTidyCheck;
  virtual ~MissingActionCheck() = default;

  void Leave(const parser::FileUnitNumber &) override;
  void Leave(const parser::OpenStmt &) override;
};

} // namespace Fortran::tidy::bugprone

#endif // LLVM_FLANG_TOOLS_FLANG_TIDY_BUGPRONE_MISSINGACTIONCHECK_H
