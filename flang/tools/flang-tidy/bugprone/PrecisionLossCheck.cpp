//===--- PrecisionLossCheck.cpp - flang-tidy ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PrecisionLossCheck.h"
#include "flang/Evaluate/expression.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include <clang/Basic/SourceLocation.h>

namespace Fortran::tidy::bugprone {

static bool IsLossOfPrecision(const semantics::SomeExpr *lhs,
                              const semantics::SomeExpr *rhs) {

  const auto &lhsType{lhs->GetType()};
  const auto &rhsType{rhs->GetType()};

  if (!lhsType || !rhsType)
    return false;

  auto lhsCat = lhsType->category();
  auto rhsCat = lhsType->category();

  // ignore derived types
  if (lhsCat == common::TypeCategory::Derived ||
      rhsCat == common::TypeCategory::Derived)
    return false;

  int lhsKind = lhsType->kind();
  int rhsKind = rhsType->kind();

  // integer -> integer, real, complex
  // real -> integer, real, complex
  // complex -> integer, real, complex
  //
  if (lhsCat == rhsCat && lhsKind < rhsKind)
    return true;

  if (lhsCat == common::TypeCategory::Complex &&
      rhsCat == common::TypeCategory::Real) {
    if (lhsKind < rhsKind)
      return true;
    return false;
  }

  if ((lhsCat == common::TypeCategory::Real ||
       lhsCat == common::TypeCategory::Integer) &&
      rhsCat == common::TypeCategory::Complex) {
    return true;
  }

  if (lhsCat == common::TypeCategory::Integer &&
      (rhsCat == common::TypeCategory::Real ||
       rhsCat == common::TypeCategory::Complex)) {
    return true;
  }

  if ((lhsCat == common::TypeCategory::Real ||
       lhsCat == common::TypeCategory::Complex) &&
      rhsCat == common::TypeCategory::Integer && lhsKind <= rhsKind) {
    return true;
  }

  return false;
}

using namespace parser::literals;
void PrecisionLossCheck::Enter(const parser::AssignmentStmt &assignment) {
  const auto &var{std::get<parser::Variable>(assignment.t)};
  const auto &expr{std::get<parser::Expr>(assignment.t)};
  const auto *lhs{semantics::GetExpr(context()->getSemanticsContext(), var)};
  const auto *rhs{semantics::GetExpr(context()->getSemanticsContext(), expr)};

  if (!lhs || !rhs)
    return;

  if (IsLossOfPrecision(lhs, rhs)) {
    Say(context()->getSemanticsContext().location().value(),
        "Possible loss of precision in implicit conversion (%s to %s) "_warn_en_US,
        rhs->GetType()->AsFortran(), lhs->GetType()->AsFortran());
  }
}

} // namespace Fortran::tidy::bugprone
