//===--- FunctionCognitiveComplexityCheck.cpp - flang-tidy ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FunctionCognitiveComplexityCheck.h"
#include "flang/Parser/parse-tree.h"

namespace Fortran::tidy::readability {

using namespace parser::literals;

FunctionCognitiveComplexityCheck::FunctionCognitiveComplexityCheck(
    llvm::StringRef name, FlangTidyContext *context)
    : FlangTidyCheck(name, context),
      Threshold(Options.get("Threshold", DefaultThreshold)) {}

void FunctionCognitiveComplexityCheck::storeOptions(
    FlangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "Threshold", Threshold);
}

void FunctionCognitiveComplexityCheck::Enter(
    const parser::SubroutineSubprogram &program) {
  currentNestingLevel_ = -1; // start at -1 to account for the first block
  maxNestingLevel_ = 0;
  inProcedure_ = true;
  cognitiveComplexity_ = 0;

  currentProcLoc_ =
      std::get<parser::Statement<parser::SubroutineStmt>>(program.t).source;
}

void FunctionCognitiveComplexityCheck::Leave(
    const parser::SubroutineSubprogram &) {
  CheckCognitiveComplexity();
  inProcedure_ = false;
}

void FunctionCognitiveComplexityCheck::Enter(
    const parser::FunctionSubprogram &stmt) {
  currentNestingLevel_ = -1;
  maxNestingLevel_ = 0;
  inProcedure_ = true;
  cognitiveComplexity_ = 0;

  currentProcLoc_ =
      std::get<parser::Statement<parser::FunctionStmt>>(stmt.t).source;
}

void FunctionCognitiveComplexityCheck::Leave(
    const parser::FunctionSubprogram &) {
  CheckCognitiveComplexity();
  inProcedure_ = false;
}

void FunctionCognitiveComplexityCheck::Enter(const parser::DoConstruct &) {
  cognitiveComplexity_ += (1 + currentNestingLevel_);
}
void FunctionCognitiveComplexityCheck::Enter(
    const parser::IfConstruct &construct) {
  cognitiveComplexity_ += (1 + currentNestingLevel_);

  // for each if-then-else, increment the cognitive complexity
  const auto &elseIf =
      std::get<std::list<parser::IfConstruct::ElseIfBlock>>(construct.t);

  cognitiveComplexity_ +=
      (elseIf.size() * currentNestingLevel_ + elseIf.size());

  // if there is an else, increment the cognitive complexity by 1 and the
  // nesting level
  if (std::get<std::optional<parser::IfConstruct::ElseBlock>>(construct.t)) {
    cognitiveComplexity_ += (1 + currentNestingLevel_);
  }
}

void FunctionCognitiveComplexityCheck::Enter(const parser::GotoStmt &) {
  ++cognitiveComplexity_;
}

void FunctionCognitiveComplexityCheck::Enter(const parser::ComputedGotoStmt &) {
  ++cognitiveComplexity_;
}

void FunctionCognitiveComplexityCheck::Enter(const parser::AssignedGotoStmt &) {
  ++cognitiveComplexity_;
}

void FunctionCognitiveComplexityCheck::Enter(const parser::CaseConstruct &) {
  cognitiveComplexity_ += (1 + currentNestingLevel_);
}

void FunctionCognitiveComplexityCheck::Enter(
    const parser::SelectRankConstruct &) {
  cognitiveComplexity_ += (1 + currentNestingLevel_);
}

void FunctionCognitiveComplexityCheck::Enter(
    const parser::SelectTypeConstruct &) {
  cognitiveComplexity_ += (1 + currentNestingLevel_);
}

void FunctionCognitiveComplexityCheck::Enter(const parser::Expr::AND &) {
  cognitiveComplexity_ += (1 + currentNestingLevel_);
}

void FunctionCognitiveComplexityCheck::Enter(const parser::Expr::OR &) {
  cognitiveComplexity_ += (1 + currentNestingLevel_);
}

void FunctionCognitiveComplexityCheck::Enter(
    const parser::AssociateConstruct &) {
  cognitiveComplexity_ += (1 + currentNestingLevel_);
}

void FunctionCognitiveComplexityCheck::Enter(const parser::Block &) {
  if (inProcedure_) {
    currentNestingLevel_++;
    UpdateMaxNestingLevel();
  }
}
void FunctionCognitiveComplexityCheck::Leave(const parser::Block &) {
  if (inProcedure_ && currentNestingLevel_ > 0) {
    currentNestingLevel_--;
  }
}

void FunctionCognitiveComplexityCheck::UpdateMaxNestingLevel() {
  maxNestingLevel_ = std::max(maxNestingLevel_, currentNestingLevel_);
}

void FunctionCognitiveComplexityCheck::CheckCognitiveComplexity() {
  if (Threshold && cognitiveComplexity_ > Threshold) {
    Say(currentProcLoc_,
        "%s has a cognitive complexity of %d, which exceeds the threshold of %d"_warn_en_US,
        currentProcLoc_, cognitiveComplexity_, Threshold.value());
  }
}

} // namespace Fortran::tidy::readability
