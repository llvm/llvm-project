//===--- FunctionCognitiveComplexityCheck.h - flang-tidy --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_TOOLS_FLANG_TIDY_READABILITY_FUNCTIONCOGNITIVECOMPLEXITYCHECK_H
#define LLVM_FLANG_TOOLS_FLANG_TIDY_READABILITY_FUNCTIONCOGNITIVECOMPLEXITYCHECK_H

#include "../FlangTidyCheck.h"
#include "flang/Parser/parse-tree.h"

namespace Fortran::tidy::readability {

/// This check verifies that the cognitive complexity of a function is
/// not too high.
///
/// For the user-facing documentation see:
/// https://flang.llvm.org/@PLACEHOLDER@/function-cognitive-complexity.html
class FunctionCognitiveComplexityCheck : public virtual FlangTidyCheck {
public:
  explicit FunctionCognitiveComplexityCheck(llvm::StringRef name,
                                            FlangTidyContext *context);
  virtual ~FunctionCognitiveComplexityCheck() = default;

  void storeOptions(FlangTidyOptions::OptionMap &Opts) override;

  void Enter(const parser::SubroutineSubprogram &) override;
  void Leave(const parser::SubroutineSubprogram &) override;
  void Enter(const parser::FunctionSubprogram &) override;
  void Leave(const parser::FunctionSubprogram &) override;

  // DoConstruct, IfConstruct & CaseConstruct
  void Enter(const parser::DoConstruct &) override;
  void Enter(const parser::IfConstruct &) override;
  void Enter(const parser::CaseConstruct &) override;

  // Boolean operators &&, ||
  void Enter(const parser::Expr::AND &) override;
  void Enter(const parser::Expr::OR &) override;

  // Select Rank? Select Type? Associate?
  void Enter(const parser::SelectRankConstruct &) override;
  void Enter(const parser::SelectTypeConstruct &) override;
  void Enter(const parser::AssociateConstruct &) override;

  // Goto statements
  void Enter(const parser::GotoStmt &) override;
  void Enter(const parser::ComputedGotoStmt &) override;
  void Enter(const parser::AssignedGotoStmt &) override;

  void Enter(const parser::Block &) override;
  void Leave(const parser::Block &) override;

private:
  const std::optional<unsigned> Threshold;

  constexpr static unsigned DefaultThreshold = 25U;

  int maxNestingLevel_;
  int currentNestingLevel_;
  int cognitiveComplexity_;
  parser::CharBlock currentProcLoc_;
  bool inProcedure_;
  void UpdateMaxNestingLevel();
  void CheckCognitiveComplexity();
};

} // namespace Fortran::tidy::readability

#endif // LLVM_FLANG_TOOLS_FLANG_TIDY_READABILITY_FUNCTIONCOGNITIVECOMPLEXITYCHECK_H
