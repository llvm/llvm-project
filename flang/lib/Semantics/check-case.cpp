//===-- lib/Semantics/check-case.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-case.h"
#include "flang/Common/idioms.h"
#include "flang/Common/reference.h"
#include "flang/Common/template.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/type.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/tools.h"
#include <tuple>

namespace Fortran::semantics {

template <typename T> class CaseValues {
public:
  CaseValues(SemanticsContext &c, const evaluate::DynamicType &t)
      : context_{c}, caseExprType_{t} {}

  void Check(const std::list<parser::CaseConstruct::Case> &cases) {
    for (const parser::CaseConstruct::Case &c : cases) {
      AddCase(c);
    }
    if (!hasErrors_) {
      cases_.sort(Comparator{});
      if (!AreCasesDisjoint()) { // C1149
        ReportConflictingCases();
      }
    }
  }

private:
  using Value = evaluate::Scalar<T>;

  void AddCase(const parser::CaseConstruct::Case &c) {
    const auto &stmt{std::get<parser::Statement<parser::CaseStmt>>(c.t)};
    const parser::CaseStmt &caseStmt{stmt.statement};
    const auto &selector{std::get<parser::CaseSelector>(caseStmt.t)};
    common::visit(
        common::visitors{
            [&](const std::list<parser::CaseValueRange> &ranges) {
              for (const auto &range : ranges) {
                auto pair{ComputeBounds(range)};
                if (pair.first && pair.second && *pair.first > *pair.second) {
                  context_.Warn(common::UsageWarning::EmptyCase, stmt.source,
                      "CASE has lower bound greater than upper bound"_warn_en_US);
                } else {
                  if constexpr (T::category == TypeCategory::Logical) { // C1148
                    if ((pair.first || pair.second) &&
                        (!pair.first || !pair.second ||
                            *pair.first != *pair.second)) {
                      context_.Say(stmt.source,
                          "CASE range is not allowed for LOGICAL"_err_en_US);
                    }
                  }
                  cases_.emplace_back(stmt);
                  cases_.back().lower = std::move(pair.first);
                  cases_.back().upper = std::move(pair.second);
                }
              }
            },
            [&](const parser::Default &) { cases_.emplace_front(stmt); },
        },
        selector.u);
  }

  std::optional<Value> GetValue(const parser::CaseValue &caseValue) {
    const auto &expr{parser::UnwrapRef<parser::Expr>(caseValue)};
    auto *x{expr.typedExpr.get()};
    if (x && x->v) { // C1147
      auto type{x->v->GetType()};
      if (type && type->category() == caseExprType_.category() &&
          (type->category() != TypeCategory::Character ||
              type->kind() == caseExprType_.kind())) {
        parser::Messages buffer; // discarded folding messages
        parser::ContextualMessages foldingMessages{expr.source, &buffer};
        evaluate::FoldingContext foldingContext{
            context_.foldingContext(), foldingMessages};
        auto folded{evaluate::Fold(foldingContext, SomeExpr{*x->v})};
        if (auto converted{evaluate::Fold(foldingContext,
                evaluate::ConvertToType(T::GetType(), SomeExpr{folded}))}) {
          if (auto value{evaluate::GetScalarConstantValue<T>(*converted)}) {
            auto back{evaluate::Fold(foldingContext,
                evaluate::ConvertToType(*type, SomeExpr{*converted}))};
            if (back == folded) {
              x->v = converted;
              return value;
            } else {
              context_.Warn(common::UsageWarning::CaseOverflow, expr.source,
                  "CASE value (%s) overflows type (%s) of SELECT CASE expression"_warn_en_US,
                  folded.AsFortran(), caseExprType_.AsFortran());
              hasErrors_ = true;
              return std::nullopt;
            }
          }
        }
        context_.Say(expr.source,
            "CASE value (%s) must be a constant scalar"_err_en_US,
            x->v->AsFortran());
      } else {
        std::string typeStr{type ? type->AsFortran() : "typeless"s};
        context_.Say(expr.source,
            "CASE value has type '%s' which is not compatible with the SELECT CASE expression's type '%s'"_err_en_US,
            typeStr, caseExprType_.AsFortran());
      }
      hasErrors_ = true;
    }
    return std::nullopt;
  }

  using PairOfValues = std::pair<std::optional<Value>, std::optional<Value>>;
  PairOfValues ComputeBounds(const parser::CaseValueRange &range) {
    return common::visit(
        common::visitors{
            [&](const parser::CaseValue &x) {
              auto value{GetValue(x)};
              return PairOfValues{value, value};
            },
            [&](const parser::CaseValueRange::Range &x) {
              std::optional<Value> lo, hi;
              if (x.lower) {
                lo = GetValue(*x.lower);
              }
              if (x.upper) {
                hi = GetValue(*x.upper);
              }
              if ((x.lower && !lo) || (x.upper && !hi)) {
                return PairOfValues{}; // error case
              }
              return PairOfValues{std::move(lo), std::move(hi)};
            },
        },
        range.u);
  }

  struct Case {
    explicit Case(const parser::Statement<parser::CaseStmt> &s) : stmt{s} {}
    bool IsDefault() const { return !lower && !upper; }
    std::string AsFortran() const {
      std::string result;
      {
        llvm::raw_string_ostream bs{result};
        if (lower) {
          evaluate::Constant<T>{*lower}.AsFortran(bs << '(');
          if (!upper) {
            bs << ':';
          } else if (*lower != *upper) {
            evaluate::Constant<T>{*upper}.AsFortran(bs << ':');
          }
          bs << ')';
        } else if (upper) {
          evaluate::Constant<T>{*upper}.AsFortran(bs << "(:") << ')';
        } else {
          bs << "DEFAULT";
        }
      }
      return result;
    }

    const parser::Statement<parser::CaseStmt> &stmt;
    std::optional<Value> lower, upper;
  };

  // Defines a comparator for use with std::list<>::sort().
  // Returns true if and only if the highest value in range x is less
  // than the least value in range y.  The DEFAULT case is arbitrarily
  // defined to be less than all others.  When two ranges overlap,
  // neither is less than the other.
  struct Comparator {
    bool operator()(const Case &x, const Case &y) const {
      if (x.IsDefault()) {
        return !y.IsDefault();
      } else {
        return x.upper && y.lower && *x.upper < *y.lower;
      }
    }
  };

  bool AreCasesDisjoint() const {
    auto endIter{cases_.end()};
    for (auto iter{cases_.begin()}; iter != endIter; ++iter) {
      auto next{iter};
      if (++next != endIter && !Comparator{}(*iter, *next)) {
        return false;
      }
    }
    return true;
  }

  // This has quadratic time, but only runs in error cases
  void ReportConflictingCases() {
    for (auto iter{cases_.begin()}; iter != cases_.end(); ++iter) {
      parser::Message *msg{nullptr};
      for (auto p{cases_.begin()}; p != cases_.end(); ++p) {
        if (p->stmt.source.begin() < iter->stmt.source.begin() &&
            !Comparator{}(*p, *iter) && !Comparator{}(*iter, *p)) {
          if (!msg) {
            msg = &context_.Say(iter->stmt.source,
                "CASE %s conflicts with previous cases"_err_en_US,
                iter->AsFortran());
          }
          msg->Attach(
              p->stmt.source, "Conflicting CASE %s"_en_US, p->AsFortran());
        }
      }
    }
  }

  SemanticsContext &context_;
  const evaluate::DynamicType &caseExprType_;
  std::list<Case> cases_;
  bool hasErrors_{false};
};

template <TypeCategory CAT> struct TypeVisitor {
  using Result = bool;
  using Types = evaluate::CategoryTypes<CAT>;
  template <typename T> Result Test() {
    if (T::kind == exprType.kind()) {
      CaseValues<T>(context, exprType).Check(caseList);
      return true;
    } else {
      return false;
    }
  }
  SemanticsContext &context;
  const evaluate::DynamicType &exprType;
  const std::list<parser::CaseConstruct::Case> &caseList;
};

void CaseChecker::Enter(const parser::CaseConstruct &construct) {
  const auto &selectCaseStmt{
      std::get<parser::Statement<parser::SelectCaseStmt>>(construct.t)};
  const auto &selectCase{selectCaseStmt.statement};
  const auto &selectExpr{
      std::get<parser::Scalar<parser::Expr>>(selectCase.t).thing};
  const auto *x{GetExpr(context_, selectExpr)};
  if (!x) {
    return; // expression semantics failed
  }
  if (auto exprType{x->GetType()}) {
    const auto &caseList{
        std::get<std::list<parser::CaseConstruct::Case>>(construct.t)};
    switch (exprType->category()) {
    case TypeCategory::Integer:
      common::SearchTypes(
          TypeVisitor<TypeCategory::Integer>{context_, *exprType, caseList});
      return;
    case TypeCategory::Unsigned:
      common::SearchTypes(
          TypeVisitor<TypeCategory::Unsigned>{context_, *exprType, caseList});
      return;
    case TypeCategory::Logical:
      CaseValues<evaluate::Type<TypeCategory::Logical, 1>>{context_, *exprType}
          .Check(caseList);
      return;
    case TypeCategory::Character:
      common::SearchTypes(
          TypeVisitor<TypeCategory::Character>{context_, *exprType, caseList});
      return;
    default:
      break;
    }
  }
  context_.Say(selectExpr.source,
      context_.IsEnabled(common::LanguageFeature::Unsigned)
          ? "SELECT CASE expression must be integer, unsigned, logical, or character"_err_en_US
          : "SELECT CASE expression must be integer, logical, or character"_err_en_US);
}
} // namespace Fortran::semantics
