//===--- UndeclaredProcCheck.cpp - flang-tidy -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UndeclaredProcCheck.h"
#include "flang/Semantics/attr.h"
#include "flang/Semantics/symbol.h"

namespace Fortran::tidy::bugprone {

using namespace parser::literals;
void UndeclaredProcCheck::CheckForUndeclaredProcedures(
    semantics::SemanticsContext &context, const semantics::Scope &scope) {
  if (scope.IsModuleFile())
    return;

  for (const auto &pair : scope) {
    const semantics::Symbol &symbol = *pair.second;
    if (auto *details{symbol.detailsIf<semantics::ProcEntityDetails>()};
        details) {
      if (symbol.owner().IsGlobal()) { // unknown global procedure
        Say(symbol.name(), "Implicit declaration of procedure '%s'"_warn_en_US,
            symbol.name());
      } else if (!details->HasExplicitInterface() && // no explicit interface
                 !symbol.attrs().test(
                     semantics::Attr::INTRINSIC)) { // not an intrinsic
        Say(symbol.name(),
            "Procedure '%s' has no explicit interface"_warn_en_US,
            symbol.name());
      }
    }
  }

  for (const semantics::Scope &child : scope.children()) {
    CheckForUndeclaredProcedures(context, child);
  }
}

UndeclaredProcCheck::UndeclaredProcCheck(llvm::StringRef name,
                                         FlangTidyContext *context)
    : FlangTidyCheck{name, context} {
  CheckForUndeclaredProcedures(context->getSemanticsContext(),
                               context->getSemanticsContext().globalScope());
}

} // namespace Fortran::tidy::bugprone
