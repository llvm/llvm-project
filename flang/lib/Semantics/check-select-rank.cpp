//===-- lib/Semantics/check-select-rank.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-select-rank.h"
#include "flang/Common/Fortran.h"
#include "flang/Common/idioms.h"
#include "flang/Parser/message.h"
#include "flang/Parser/tools.h"
#include "flang/Semantics/tools.h"
#include <list>
#include <optional>
#include <set>
#include <tuple>
#include <variant>

namespace Fortran::semantics {

void SelectRankConstructChecker::Leave(
    const parser::SelectRankConstruct &selectRankConstruct) {
  const auto &selectRankStmt{
      std::get<parser::Statement<parser::SelectRankStmt>>(
          selectRankConstruct.t)};
  const auto &selectRankStmtSel{
      std::get<parser::Selector>(selectRankStmt.statement.t)};

  // R1149 select-rank-stmt checks
  const Symbol *saveSelSymbol{nullptr};
  if (const auto selExpr{GetExprFromSelector(selectRankStmtSel)}) {
    if (const Symbol * sel{evaluate::UnwrapWholeSymbolDataRef(*selExpr)}) {
      if (!evaluate::IsAssumedRank(*sel)) { // C1150
        context_.Say(parser::FindSourceLocation(selectRankStmtSel),
            "Selector '%s' is not an assumed-rank array variable"_err_en_US,
            sel->name().ToString());
      } else {
        saveSelSymbol = sel;
      }
    } else {
      context_.Say(parser::FindSourceLocation(selectRankStmtSel),
          "Selector '%s' is not an assumed-rank array variable"_err_en_US,
          parser::FindSourceLocation(selectRankStmtSel).ToString());
    }
  }

  // R1150 select-rank-case-stmt checks
  auto &rankCaseList{std::get<std::list<parser::SelectRankConstruct::RankCase>>(
      selectRankConstruct.t)};
  bool defaultRankFound{false};
  bool starRankFound{false};
  parser::CharBlock prevLocDefault;
  parser::CharBlock prevLocStar;
  std::optional<parser::CharBlock> caseForRank[common::maxRank + 1];

  for (const auto &rankCase : rankCaseList) {
    const auto &rankCaseStmt{
        std::get<parser::Statement<parser::SelectRankCaseStmt>>(rankCase.t)};
    const auto &rank{
        std::get<parser::SelectRankCaseStmt::Rank>(rankCaseStmt.statement.t)};
    common::visit(
        common::visitors{
            [&](const parser::Default &) { // C1153
              if (!defaultRankFound) {
                defaultRankFound = true;
                prevLocDefault = rankCaseStmt.source;
              } else {
                context_
                    .Say(rankCaseStmt.source,
                        "Not more than one of the selectors of SELECT RANK "
                        "statement may be DEFAULT"_err_en_US)
                    .Attach(prevLocDefault, "Previous use"_en_US);
              }
            },
            [&](const parser::Star &) { // C1153
              if (!starRankFound) {
                starRankFound = true;
                prevLocStar = rankCaseStmt.source;
              } else {
                context_
                    .Say(rankCaseStmt.source,
                        "Not more than one of the selectors of SELECT RANK "
                        "statement may be '*'"_err_en_US)
                    .Attach(prevLocStar, "Previous use"_en_US);
              }
              if (saveSelSymbol &&
                  IsAllocatableOrPointer(*saveSelSymbol)) { // C1155
                context_.Say(parser::FindSourceLocation(selectRankStmtSel),
                    "RANK (*) cannot be used when selector is "
                    "POINTER or ALLOCATABLE"_err_en_US);
              }
            },
            [&](const parser::ScalarIntConstantExpr &init) {
              if (auto val{GetIntValue(init)}) {
                // If value is in valid range, then only show
                // value repeat error, else stack smashing occurs
                if (*val < 0 || *val > common::maxRank) { // C1151
                  context_.Say(rankCaseStmt.source,
                      "The value of the selector must be "
                      "between zero and %d"_err_en_US,
                      common::maxRank);

                } else {
                  if (!caseForRank[*val].has_value()) {
                    caseForRank[*val] = rankCaseStmt.source;
                  } else {
                    auto prevloc{caseForRank[*val].value()};
                    context_
                        .Say(rankCaseStmt.source,
                            "Same rank value (%d) not allowed more than once"_err_en_US,
                            *val)
                        .Attach(prevloc, "Previous use"_en_US);
                  }
                }
              }
            },
        },
        rank.u);
  }
}

const SomeExpr *SelectRankConstructChecker::GetExprFromSelector(
    const parser::Selector &selector) {
  return common::visit([](const auto &x) { return GetExpr(x); }, selector.u);
}

} // namespace Fortran::semantics
