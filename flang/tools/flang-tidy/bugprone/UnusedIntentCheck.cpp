//===--- UnusedIntentCheck.cpp - flang-tidy -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnusedIntentCheck.h"
#include "flang/Semantics/attr.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"

namespace Fortran::tidy::bugprone {

using namespace parser::literals;

static std::unordered_map<const semantics::Symbol *, const semantics::Symbol *>
    procBindingDetailsSymbolsMap;

void UnusedIntentCheck::CheckUnusedIntentHelper(
    semantics::SemanticsContext &context, const semantics::Scope &scope) {
  if (scope.IsModuleFile())
    return;

  auto WasDefined{[&context](const semantics::Symbol &symbol) {
    return context.IsSymbolDefined(symbol) ||
           semantics::IsInitialized(symbol, false, false, false);
  }};
  for (const auto &pair : scope) {
    const semantics::Symbol &symbol = *pair.second;
    if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()};
        details && details->isDummy()) {
      const auto &owningProcScope = symbol.owner();
      const auto &owningProc = owningProcScope.symbol();

      if (procBindingDetailsSymbolsMap.find(owningProc) !=
          procBindingDetailsSymbolsMap.end()) {
        continue;
      }
      if (!WasDefined(symbol) && semantics::IsIntentInOut(symbol)) {
        Say(symbol.name(),
            "Dummy argument '%s' with intent(inout) is never written to, consider changing to intent(in)"_warn_en_US,
            symbol.name());
      }
      if (!symbol.attrs().HasAny(
              {semantics::Attr::INTENT_IN, semantics::Attr::INTENT_INOUT,
               semantics::Attr::INTENT_OUT, semantics::Attr::VALUE})) {
        // warn about dummy arguments without explicit intent
        Say(symbol.name(),
            "Dummy argument '%s' has no explicit intent"_warn_en_US,
            symbol.name());
      }
    }
  }

  for (const semantics::Scope &child : scope.children()) {
    CheckUnusedIntentHelper(context, child);
  }
}

static void MakeProcBindingSymbolSet(semantics::SemanticsContext &context,
                                     const semantics::Scope &scope) {
  for (const auto &pair : scope) {
    const semantics::Symbol &symbol = *pair.second;
    if (auto *details{symbol.detailsIf<semantics::ProcBindingDetails>()}) {
      procBindingDetailsSymbolsMap[&details->symbol()] = &symbol;
    }
  }

  for (const semantics::Scope &child : scope.children()) {
    MakeProcBindingSymbolSet(context, child);
  }
}

UnusedIntentCheck::UnusedIntentCheck(llvm::StringRef name,
                                     FlangTidyContext *context)
    : FlangTidyCheck{name, context} {

  MakeProcBindingSymbolSet(context->getSemanticsContext(),
                           context->getSemanticsContext().globalScope());

  CheckUnusedIntentHelper(context->getSemanticsContext(),
                          context->getSemanticsContext().globalScope());
}

} // namespace Fortran::tidy::bugprone
