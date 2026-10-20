//===--- ArithmeticIfStmtCheck.cpp - flang-tidy ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ArithmeticIfStmtCheck.h"

namespace Fortran::tidy::bugprone {

using namespace parser::literals;
void ArithmeticIfStmtCheck::Enter(const parser::ArithmeticIfStmt &ifStmt) {
  if (context()->getSemanticsContext().location().has_value()) {
    Say(context()->getSemanticsContext().location().value(),
        "Arithmetic if statements are not recommended"_warn_en_US);
  }
}

} // namespace Fortran::tidy::bugprone
