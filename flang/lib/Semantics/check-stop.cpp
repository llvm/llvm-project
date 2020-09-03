//===-- lib/Semantics/check-stop.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-stop.h"
#include "flang/Common/Fortran.h"
#include "flang/Evaluate/expression.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/tools.h"
#include <optional>

namespace Fortran::semantics {

void StopChecker::Enter(const parser::StopStmt &stmt) {
  const auto &stopCode{std::get<std::optional<parser::StopCode>>(stmt.t)};
  if (const auto *expr{GetExpr(stopCode)}) {
    const parser::CharBlock &source{parser::FindSourceLocation(stopCode)};
    if (ExprHasTypeCategory(*expr, common::TypeCategory::Integer)) {
      // C1171 default kind
      if (!ExprTypeKindIsDefault(*expr, context_)) {
        context_.Say(
            source, "INTEGER stop code must be of default kind"_err_en_US);
      }
    } else if (ExprHasTypeCategory(*expr, common::TypeCategory::Character)) {
      // R1162 spells scalar-DEFAULT-char-expr
      if (!ExprTypeKindIsDefault(*expr, context_)) {
        context_.Say(
            source, "CHARACTER stop code must be of default kind"_err_en_US);
      }
    } else {
      context_.Say(
          source, "Stop code must be of INTEGER or CHARACTER type"_err_en_US);
    }
  }
}

} // namespace Fortran::semantics
