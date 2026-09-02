//===--- ImplicitDeclCheck.cpp - flang-tidy -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImplicitDeclCheck.h"

namespace Fortran::tidy::bugprone {

using namespace parser::literals;
void ImplicitDeclCheck::CheckForImplicitDeclarations(
    semantics::SemanticsContext &context, const semantics::Scope &scope) {
  if (scope.IsModuleFile())
    return;

  for (const auto &pair : scope) {
    const semantics::Symbol &symbol = *pair.second;
    if (symbol.test(semantics::Symbol::Flag::Implicit) &&
        !symbol.test(semantics::Symbol::Flag::Function) &&
        !symbol.test(semantics::Symbol::Flag::Subroutine)) {
      Say(symbol.name(), "Implicit declaration of symbol '%s'"_warn_en_US,
          symbol.name());
    }
  }

  for (const semantics::Scope &child : scope.children()) {
    CheckForImplicitDeclarations(context, child);
  }
}

ImplicitDeclCheck::ImplicitDeclCheck(llvm::StringRef name,
                                     FlangTidyContext *context)
    : FlangTidyCheck{name, context} {
  CheckForImplicitDeclarations(context->getSemanticsContext(),
                               context->getSemanticsContext().globalScope());
}

} // namespace Fortran::tidy::bugprone
