//===-- lib/Semantics/check-warning.cpp
//-----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-warning.h"
#include "flang/Semantics/tools.h"

namespace Fortran::semantics {

void WarningChecker::Enter(const parser::FunctionStmt &stmt) {
  if (Wunused_dummy_argument) {
    for (const auto &dummyName : std::get<std::list<parser::Name>>(stmt.t)) {
      if (auto *detail = dummyName.symbol->detailsIf<ObjectEntityDetails>()) {
        const Symbol *ownerSymbol{dummyName.symbol->owner().symbol()};
        const auto *ownerSubp{ownerSymbol->detailsIf<SubprogramDetails>()};
        bool inInterface{ownerSubp && ownerSubp->isInterface()};

        if (!inInterface && !detail->isUsed()) {
          context_.Say(dummyName.symbol->GetUltimate().name(),
              "Unused dummy argument '%s' [-Wunused-dummy-argument]"_warn_en_US,
              dummyName.ToString());
        }
      }
    }
  }
  if (Wunused_variable) {
    if (const auto &suffix{std::get<std::optional<parser::Suffix>>(stmt.t)}) {
      if (suffix->resultName.has_value()) {
        if (auto *detail =
                suffix->resultName->symbol->detailsIf<ObjectEntityDetails>()) {
          const Symbol *ownerSymbol{
              suffix->resultName->symbol->owner().symbol()};
          const auto *ownerSubp{ownerSymbol->detailsIf<SubprogramDetails>()};
          bool inInterface{ownerSubp && ownerSubp->isInterface()};
          if (!inInterface && !detail->isUsed()) {
            context_.Say(suffix->resultName->source,
                "Unused variable '%s' [-Wunused-variable]"_warn_en_US,
                suffix->resultName->ToString());
          }
        }
      }
    }
  }
}

void WarningChecker::Enter(const parser::SubroutineStmt &stmt) {
  if (!Wunused_dummy_argument) {
    return;
  }
  for (const auto &dummyArg : std::get<std::list<parser::DummyArg>>(stmt.t)) {
    if (const auto *dummyName{std::get_if<parser::Name>(&dummyArg.u)}) {
      if (const auto *symbol = dummyName->symbol) {

        const Symbol *ownerSymbol{symbol->owner().symbol()};
        const auto *ownerSubp{ownerSymbol->detailsIf<SubprogramDetails>()};
        bool inInterface{ownerSubp && ownerSubp->isInterface()};

        if (auto *detail = symbol->detailsIf<ObjectEntityDetails>()) {
          if (!inInterface && !detail->isUsed()) {
            context_.Say(symbol->GetUltimate().name(),
                "Unused dummy argument '%s' [-Wunused-dummy-argument]"_warn_en_US,
                dummyName->ToString());
          }
        }
      }
    }
  }
}

void WarningChecker::Enter(const parser::EntityDecl &decl) {
  if (!Wunused_variable) {
    return;
  }

  const auto &name{std::get<parser::ObjectName>(decl.t)};
  if (const auto *symbol = name.symbol) {
    const Symbol *ownerSymbol{symbol->owner().symbol()};
    const auto *ownerSubp{ownerSymbol->detailsIf<SubprogramDetails>()};
    bool inInterface{ownerSubp && ownerSubp->isInterface()};
    bool inModule{ownerSymbol && ownerSymbol->scope() &&
        ownerSymbol->scope()->IsModule()};

    if (auto *detail = symbol->detailsIf<ObjectEntityDetails>()) {
      if (!inInterface && !inModule && !detail->isDummy() &&
          !detail->isFuncResult() && !detail->isUsed()) {
        context_.Say(symbol->name(),
            "Unused variable '%s' [-Wunused-variable]"_warn_en_US,
            name.ToString());
      }
    }
  }
}

} // namespace Fortran::semantics
