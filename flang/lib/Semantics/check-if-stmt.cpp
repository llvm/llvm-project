//===-- lib/Semantics/check-if-stmt.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-if-stmt.h"
#include "flang/Parser/message.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"

namespace Fortran::semantics {

void IfStmtChecker::Leave(const parser::IfStmt &ifStmt) {
  // C1143 Check that the action stmt is not an if stmt
  const auto &body{
      std::get<parser::UnlabeledStatement<parser::ActionStmt>>(ifStmt.t)};
  if (std::holds_alternative<common::Indirection<parser::IfStmt>>(
          body.statement.u)) {
    context_.Say(
        body.source, "IF statement is not allowed in IF statement"_err_en_US);
  }
}

} // namespace Fortran::semantics
