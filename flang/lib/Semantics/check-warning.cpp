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
#include "clang/Basic/AllDiagnostics.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Basic/DiagnosticOptions.h"

namespace Fortran::semantics {

void WarningChecker::Enter(const parser::FunctionStmt &stmt) {
  clang::DiagnosticsEngine &diags = context_.getDiagnostics();
  auto opts = diags.getDiagnosticOptions().Warnings;
  for (const auto &dummyName : std::get<std::list<parser::Name>>(stmt.t)) {
    if (auto *detail = dummyName.symbol->detailsIf<ObjectEntityDetails>()) {
      const Symbol *ownerSymbol{dummyName.symbol->owner().symbol()};
      const auto *ownerSubp{ownerSymbol->detailsIf<SubprogramDetails>()};
      bool inInterface{ownerSubp && ownerSubp->isInterface()};
      if (!inInterface && !detail->isUsed()) {
        diags.Report(clang::diag::warn_unused_dummy_argument)
            << dummyName.ToString();
      }
    }
  }
  if (const auto &suffix{std::get<std::optional<parser::Suffix>>(stmt.t)}) {
    if (suffix->resultName.has_value()) {
      if (auto *detail =
              suffix->resultName->symbol->detailsIf<ObjectEntityDetails>()) {
        const Symbol *ownerSymbol{suffix->resultName->symbol->owner().symbol()};
        const auto *ownerSubp{ownerSymbol->detailsIf<SubprogramDetails>()};
        bool inInterface{ownerSubp && ownerSubp->isInterface()};
        if (!inInterface && !detail->isUsed()) {
          diags.Report(clang::diag::warn_unused_variable)
              << suffix->resultName->ToString();
        }
      }
    }
  }
}

void WarningChecker::Enter(const parser::SubroutineStmt &stmt) {
  clang::DiagnosticsEngine &diags = context_.getDiagnostics();
  for (const auto &dummyArg : std::get<std::list<parser::DummyArg>>(stmt.t)) {
    if (const auto *dummyName{std::get_if<parser::Name>(&dummyArg.u)}) {
      if (const auto *symbol = dummyName->symbol) {

        const Symbol *ownerSymbol{symbol->owner().symbol()};
        const auto *ownerSubp{ownerSymbol->detailsIf<SubprogramDetails>()};
        bool inInterface{ownerSubp && ownerSubp->isInterface()};

        if (auto *detail = symbol->detailsIf<ObjectEntityDetails>()) {
          if (!inInterface && !detail->isUsed()) {
            diags.Report(clang::diag::warn_unused_dummy_argument)
                << dummyName->ToString();
          }
        }
      }
    }
  }
}

void WarningChecker::Enter(const parser::EntityDecl &decl) {
  clang::DiagnosticsEngine &diags = context_.getDiagnostics();
  const auto &name{std::get<parser::ObjectName>(decl.t)};
  if (const auto *symbol = name.symbol) {
    if (const Symbol *ownerSymbol = symbol->owner().symbol()) {
      const auto *ownerSubp{ownerSymbol->detailsIf<SubprogramDetails>()};
      bool inInterface{ownerSubp && ownerSubp->isInterface()};
      bool inModule{ownerSymbol && ownerSymbol->scope() &&
          ownerSymbol->scope()->IsModule()};

      if (auto *detail = symbol->detailsIf<ObjectEntityDetails>()) {
        if (!inInterface && !inModule && !detail->isDummy() &&
            !detail->isFuncResult() && !detail->isUsed()) {
          diags.Report(clang::diag::warn_unused_variable) << name.ToString();
        }
      }
    }
  }
}

} // namespace Fortran::semantics
