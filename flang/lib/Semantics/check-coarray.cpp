//===-- lib/Semantics/check-coarray.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-coarray.h"
#include "flang/Common/indirection.h"
#include "flang/Evaluate/expression.h"
#include "flang/Parser/message.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/tools.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/tools.h"

namespace Fortran::semantics {

class CriticalBodyEnforce {
public:
  CriticalBodyEnforce(
      SemanticsContext &context, parser::CharBlock criticalSourcePosition)
      : context_{context}, criticalSourcePosition_{criticalSourcePosition} {}
  std::set<parser::Label> labels() { return labels_; }
  template <typename T> bool Pre(const T &) { return true; }
  template <typename T> void Post(const T &) {}

  template <typename T> bool Pre(const parser::Statement<T> &statement) {
    currentStatementSourcePosition_ = statement.source;
    if (statement.label.has_value()) {
      labels_.insert(*statement.label);
    }
    return true;
  }

  // C1118
  void Post(const parser::ReturnStmt &) {
    context_
        .Say(currentStatementSourcePosition_,
            "RETURN statement is not allowed in a CRITICAL construct"_err_en_US)
        .Attach(criticalSourcePosition_, GetEnclosingMsg());
  }
  void Post(const parser::ExecutableConstruct &construct) {
    if (IsImageControlStmt(construct)) {
      context_
          .Say(currentStatementSourcePosition_,
              "An image control statement is not allowed in a CRITICAL"
              " construct"_err_en_US)
          .Attach(criticalSourcePosition_, GetEnclosingMsg());
    }
  }

private:
  parser::MessageFixedText GetEnclosingMsg() {
    return "Enclosing CRITICAL statement"_en_US;
  }

  SemanticsContext &context_;
  std::set<parser::Label> labels_;
  parser::CharBlock currentStatementSourcePosition_;
  parser::CharBlock criticalSourcePosition_;
};

template <typename T>
static void CheckTeamType(SemanticsContext &context, const T &x) {
  if (const auto *expr{GetExpr(x)}) {
    if (!IsTeamType(evaluate::GetDerivedTypeSpec(expr->GetType()))) {
      context.Say(parser::FindSourceLocation(x), // C1114
          "Team value must be of type TEAM_TYPE from module ISO_FORTRAN_ENV"_err_en_US);
    }
  }
}

static void CheckTeamStat(
    SemanticsContext &context, const parser::ImageSelectorSpec::Stat &stat) {
  const parser::Variable &var{stat.v.thing.thing.value()};
  if (parser::GetCoindexedNamedObject(var)) {
    context.Say(parser::FindSourceLocation(var), // C931
        "Image selector STAT variable must not be a coindexed "
        "object"_err_en_US);
  }
}

void CoarrayChecker::Leave(const parser::ChangeTeamStmt &x) {
  CheckNamesAreDistinct(std::get<std::list<parser::CoarrayAssociation>>(x.t));
  CheckTeamType(context_, std::get<parser::TeamValue>(x.t));
}

void CoarrayChecker::Leave(const parser::SyncTeamStmt &x) {
  CheckTeamType(context_, std::get<parser::TeamValue>(x.t));
}

void CoarrayChecker::Leave(const parser::ImageSelector &imageSelector) {
  haveStat_ = false;
  haveTeam_ = false;
  haveTeamNumber_ = false;
  for (const auto &imageSelectorSpec :
      std::get<std::list<parser::ImageSelectorSpec>>(imageSelector.t)) {
    if (const auto *team{
            std::get_if<parser::TeamValue>(&imageSelectorSpec.u)}) {
      if (haveTeam_) {
        context_.Say(parser::FindSourceLocation(imageSelectorSpec), // C929
            "TEAM value can only be specified once"_err_en_US);
      }
      CheckTeamType(context_, *team);
      haveTeam_ = true;
    }
    if (const auto *stat{std::get_if<parser::ImageSelectorSpec::Stat>(
            &imageSelectorSpec.u)}) {
      if (haveStat_) {
        context_.Say(parser::FindSourceLocation(imageSelectorSpec), // C929
            "STAT variable can only be specified once"_err_en_US);
      }
      CheckTeamStat(context_, *stat);
      haveStat_ = true;
    }
    if (std::get_if<parser::ImageSelectorSpec::Team_Number>(
            &imageSelectorSpec.u)) {
      if (haveTeamNumber_) {
        context_.Say(parser::FindSourceLocation(imageSelectorSpec), // C929
            "TEAM_NUMBER value can only be specified once"_err_en_US);
      }
      haveTeamNumber_ = true;
    }
  }
  if (haveTeam_ && haveTeamNumber_) {
    context_.Say(parser::FindSourceLocation(imageSelector), // C930
        "Cannot specify both TEAM and TEAM_NUMBER"_err_en_US);
  }
}

void CoarrayChecker::Leave(const parser::FormTeamStmt &x) {
  CheckTeamType(context_, std::get<parser::TeamVariable>(x.t));
}

void CoarrayChecker::Enter(const parser::CriticalConstruct &x) {
  auto &criticalStmt{std::get<parser::Statement<parser::CriticalStmt>>(x.t)};

  const parser::Block &block{std::get<parser::Block>(x.t)};
  CriticalBodyEnforce criticalBodyEnforce{context_, criticalStmt.source};
  parser::Walk(block, criticalBodyEnforce);

  // C1119
  LabelEnforce criticalLabelEnforce{
      context_, criticalBodyEnforce.labels(), criticalStmt.source, "CRITICAL"};
  parser::Walk(block, criticalLabelEnforce);
}

// Check that coarray names and selector names are all distinct.
void CoarrayChecker::CheckNamesAreDistinct(
    const std::list<parser::CoarrayAssociation> &list) {
  std::set<parser::CharBlock> names;
  auto getPreviousUse{
      [&](const parser::Name &name) -> const parser::CharBlock * {
        auto pair{names.insert(name.source)};
        return !pair.second ? &*pair.first : nullptr;
      }};
  for (const auto &assoc : list) {
    const auto &decl{std::get<parser::CodimensionDecl>(assoc.t)};
    const auto &selector{std::get<parser::Selector>(assoc.t)};
    const auto &declName{std::get<parser::Name>(decl.t)};
    if (context_.HasError(declName)) {
      continue; // already reported an error about this name
    }
    if (auto *prev{getPreviousUse(declName)}) {
      Say2(declName.source, // C1113
          "Coarray '%s' was already used as a selector or coarray in this statement"_err_en_US,
          *prev, "Previous use of '%s'"_en_US);
    }
    // ResolveNames verified the selector is a simple name
    const parser::Name *name{parser::Unwrap<parser::Name>(selector)};
    if (name) {
      if (auto *prev{getPreviousUse(*name)}) {
        Say2(name->source, // C1113, C1115
            "Selector '%s' was already used as a selector or coarray in this statement"_err_en_US,
            *prev, "Previous use of '%s'"_en_US);
      }
    }
  }
}

void CoarrayChecker::Say2(const parser::CharBlock &name1,
    parser::MessageFixedText &&msg1, const parser::CharBlock &name2,
    parser::MessageFixedText &&msg2) {
  context_.Say(name1, std::move(msg1), name1)
      .Attach(name2, std::move(msg2), name2);
}
} // namespace Fortran::semantics
