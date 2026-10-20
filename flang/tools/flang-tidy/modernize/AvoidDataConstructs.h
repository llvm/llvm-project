//===--- AvoidDataConstructs.h - flang-tidy ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_TOOLS_FLANG_TIDY_MODERNIZE_AVOIDDATACONSTRUCTS_H
#define LLVM_FLANG_TOOLS_FLANG_TIDY_MODERNIZE_AVOIDDATACONSTRUCTS_H

#include "../FlangTidyCheck.h"

namespace Fortran::tidy::modernize {

/// This check verifies that DATA constructs are avoided.
///
/// For the user-facing documentation see:
/// https://flang.llvm.org/@PLACEHOLDER@/avoid-data-constructs.html
class AvoidDataConstructsCheck : public virtual FlangTidyCheck {
public:
  using FlangTidyCheck::FlangTidyCheck;
  virtual ~AvoidDataConstructsCheck() = default;
  void Enter(const parser::DataStmt &) override;
};

} // namespace Fortran::tidy::modernize

#endif // LLVM_FLANG_TOOLS_FLANG_TIDY_MODERNIZE_AVOIDDATACONSTRUCTS_H
