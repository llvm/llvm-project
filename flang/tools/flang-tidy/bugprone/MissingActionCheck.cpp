//===--- MissingActionCheck.cpp - flang-tidy ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MissingActionCheck.h"
#include "flang/Evaluate/check-expression.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"

#include <algorithm>

namespace Fortran::tidy::bugprone {

using namespace parser::literals;

void MissingActionCheck::Leave(const parser::FileUnitNumber &fileUnit) {
  // warn if its a const expr
  const auto *expr =
      semantics::GetExpr(context()->getSemanticsContext(), fileUnit.v);

  if (expr && evaluate::IsConstantExpr(*expr)) {
    Say(context()->getSemanticsContext().location().value(),
        "File unit number is a constant literal"_warn_en_US);
  }
}

void MissingActionCheck::Leave(const parser::OpenStmt &openStmt) {
  const auto &source = context()->getSemanticsContext().location().value();

  const auto &connectSpec = openStmt.v;

  const auto &action = std::find_if(
      connectSpec.begin(), connectSpec.end(), [](const auto &spec) {
        return std::holds_alternative<parser::ConnectSpec::CharExpr>(spec.u) &&
               std::get<parser::ConnectSpec::CharExpr::Kind>(
                   std::get<parser::ConnectSpec::CharExpr>(spec.u).t) ==
                   parser::ConnectSpec::CharExpr::Kind::Action;
      });

  if (action == connectSpec.end()) {
    Say(source, "ACTION specifier is missing"_warn_en_US);
  }
}

} // namespace Fortran::tidy::bugprone
