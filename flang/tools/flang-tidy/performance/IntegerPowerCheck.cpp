//===--- IntegerPowerCheck.cpp - flang-tidy -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IntegerPowerCheck.h"
#include "flang/Evaluate/check-expression.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/type.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include <variant>

namespace Fortran::tidy::performance {

IntegerPowerCheck::IntegerPowerCheck(llvm::StringRef name,
                                     FlangTidyContext *context)
    : FlangTidyCheck{name, context} {}

struct RealConstantChecker {
  bool isIntegerLike = false;

  template <int KIND>
  void operator()(
      const evaluate::Expr<evaluate::Type<evaluate::TypeCategory::Real, KIND>>
          &expr) {
    using RealType = evaluate::Type<evaluate::TypeCategory::Real, KIND>;

    if (auto scalar = evaluate::GetScalarConstantValue<RealType>(expr)) {
      auto wholeResult = scalar->ToWholeNumber();

      if (!wholeResult.flags.test(evaluate::RealFlag::InvalidArgument) &&
          !wholeResult.flags.test(evaluate::RealFlag::Overflow)) {

        if (scalar->Compare(wholeResult.value) == evaluate::Relation::Equal) {
          isIntegerLike = true;
        }
      }
    }
  }

  template <typename T>
  void operator()(const T &) {}
};

void IntegerPowerCheck::Enter(const parser::Expr::Power &power) {
  const auto &[lhs, rhs] = power.t;
  const auto *expr{semantics::GetExpr(context()->getSemanticsContext(), rhs)};

  if (!expr || !evaluate::IsConstantExpr(*expr))
    return;

  if (auto *realExpr =
          std::get_if<evaluate::Expr<evaluate::SomeReal>>(&expr->u)) {
    RealConstantChecker checker;

    std::visit(checker, realExpr->u);

    if (checker.isIntegerLike) {
      Say(rhs.value().source,
          "real exponent can be written as an integer literal"_warn_en_US);
    }
  }
}

} // namespace Fortran::tidy::performance
