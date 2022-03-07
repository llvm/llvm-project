//===-- lib/Semantics/check-return.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-return.h"
#include "flang/Common/Fortran-features.h"
#include "flang/Parser/message.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/tools.h"

namespace Fortran::semantics {

static const Scope *FindContainingSubprogram(const Scope &start) {
  const Scope &scope{GetProgramUnitContaining(start)};
  return scope.kind() == Scope::Kind::MainProgram ||
          scope.kind() == Scope::Kind::Subprogram
      ? &scope
      : nullptr;
}

void ReturnStmtChecker::Leave(const parser::ReturnStmt &returnStmt) {
  // R1542 Expression analysis validates the scalar-int-expr
  // C1574 The return-stmt shall be in the inclusive scope of a function or
  // subroutine subprogram.
  // C1575 The scalar-int-expr is allowed only in the inclusive scope of a
  // subroutine subprogram.
  const auto &scope{context_.FindScope(context_.location().value())};
  if (const auto *subprogramScope{FindContainingSubprogram(scope)}) {
    if (returnStmt.v &&
        (subprogramScope->kind() == Scope::Kind::MainProgram ||
            IsFunction(*subprogramScope->GetSymbol()))) {
      context_.Say(
          "RETURN with expression is only allowed in SUBROUTINE subprogram"_err_en_US);
    } else if (context_.ShouldWarn(common::LanguageFeature::ProgramReturn)) {
      context_.Say("RETURN should not appear in a main program"_port_en_US);
    }
  }
}

} // namespace Fortran::semantics
