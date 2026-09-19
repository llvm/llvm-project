//===--- FunctionSizeCheck.h - flang-tidy -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_TOOLS_FLANG_TIDY_READABILITY_FUNCTIONSIZECHECK_H
#define LLVM_FLANG_TOOLS_FLANG_TIDY_READABILITY_FUNCTIONSIZECHECK_H

#include "../FlangTidyCheck.h"
#include "flang/Parser/char-block.h"
#include "flang/Parser/parse-tree.h"

namespace Fortran::tidy::readability {

/// This check verifies that the size of functions and subroutines is within a
/// certain threshold.
///
/// For the user-facing documentation see:
/// https://flang.llvm.org/@PLACEHOLDER@/function-size.html
class FunctionSizeCheck : public virtual FlangTidyCheck {
public:
  explicit FunctionSizeCheck(llvm::StringRef name, FlangTidyContext *context);
  virtual ~FunctionSizeCheck() = default;

  void storeOptions(FlangTidyOptions::OptionMap &Opts) override;

  void Enter(const parser::SubroutineSubprogram &) override;
  void Leave(const parser::SubroutineSubprogram &) override;
  void Enter(const parser::FunctionSubprogram &) override;
  void Leave(const parser::FunctionSubprogram &) override;

  void Enter(const parser::Block &) override;
  void Leave(const parser::Block &) override;

private:
  void UpdateMaxNestingLevel();
  void CheckNestingThreshold();

  const std::optional<unsigned> LineThreshold;
  const std::optional<unsigned> ParameterThreshold;
  const std::optional<unsigned> NestingThreshold;

  static constexpr std::optional<unsigned> DefaultLineThreshold = std::nullopt;
  static constexpr std::optional<unsigned> DefaultParameterThreshold =
      std::nullopt;
  static constexpr std::optional<unsigned> DefaultNestingThreshold =
      std::nullopt;

  bool inProcedure_ = false;
  int currentNestingLevel_ = 0;
  int maxNestingLevel_ = 0;
  parser::CharBlock currentProcLoc_;
};

/*
 * TODO: StatementThreshold, BranchThreshold, VariableThreshold
 LineThreshold(Options.get("LineThreshold", DefaultLineThreshold)),
 (
     Options.get("StatementThreshold", DefaultStatementThreshold)),
 ParameterThreshold(
     Options.get("ParameterThreshold", DefaultParameterThreshold)),
 NestingThreshold(
     Options.get("NestingThreshold", DefaultNestingThreshold)),
 *
 */

} // namespace Fortran::tidy::readability

#endif // LLVM_FLANG_TOOLS_FLANG_TIDY_READABILITY_FUNCTIONSIZECHECK_H
