//===--- UnusedUSECheck.cpp - flang-tidy ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnusedUSECheck.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include <variant>

namespace Fortran::tidy::readability {

using namespace parser::literals;

UnusedUSECheck::UnusedUSECheck(llvm::StringRef name, FlangTidyContext *context)
    : FlangTidyCheck(name, context) {}

void UnusedUSECheck::Enter(const parser::UseStmt &stmt) {
  const semantics::Symbol *moduleSymbol = stmt.moduleName.symbol;
  if (!moduleSymbol) {
    return;
  }

  if (std::holds_alternative<std::list<parser::Only>>(stmt.u)) {
    const auto &onlyList = std::get<std::list<parser::Only>>(stmt.u);
    // USE module, ONLY: symbol1, symbol2, ...
    for (const auto &only : onlyList) {
      if (const auto *name = std::get_if<parser::Name>(&only.u)) {
        if (name->symbol) {
          importedSymbols_[name->symbol] = stmt.moduleName.source;
        }
      } else if (const auto *rename = std::get_if<parser::Rename>(&only.u)) {
        const auto &names = std::get<parser::Rename::Names>(rename->u);
        const auto &localName = std::get<0>(names.t);
        if (localName.symbol) {
          importedSymbols_[localName.symbol] = stmt.moduleName.source;
        }
      }
    }
  } else {
    // std::holds_alternative<std::list<parser::Rename>>(stmt.u)
    const auto &renameList = std::get<std::list<parser::Rename>>(stmt.u);
    if (!renameList.empty()) {
      // USE module, local_name => module_name, ...
      for (const auto &rename : renameList) {
        if (std::holds_alternative<parser::Rename::Names>(rename.u)) {
          const auto &names = std::get<parser::Rename::Names>(rename.u);
          const auto &localName = std::get<0>(names.t);
          if (localName.symbol) {
            importedSymbols_[localName.symbol] = stmt.moduleName.source;
          }
        }
      }
    } else {
      // empty rename list means USE module (imports everything)
      wholeModuleImports_[moduleSymbol] = stmt.moduleName.source;
    }
  }
}

void UnusedUSECheck::Enter(const parser::Name &name) {
  if (!name.symbol) {
    return;
  }

  if (importedSymbols_.find(name.symbol) != importedSymbols_.end()) {
    usedSymbols_.insert(name.symbol);
    return;
  }

  if (const auto *details = name.symbol->detailsIf<semantics::UseDetails>()) {
    const semantics::Symbol *moduleSymbol = details->symbol().owner().symbol();
    if (wholeModuleImports_.find(moduleSymbol) != wholeModuleImports_.end()) {
      usedModules_.insert(moduleSymbol);
    }
  }
}

void UnusedUSECheck::Leave(const parser::ProgramUnit &) {
  for (const auto &[symbol, source] : importedSymbols_) {
    if (usedSymbols_.find(symbol) == usedSymbols_.end()) {
      Say(source, "Unused symbol '%s' in USE statement"_warn_en_US,
          symbol->name());
    }
  }

  for (const auto &[moduleSymbol, source] : wholeModuleImports_) {
    if (usedModules_.find(moduleSymbol) == usedModules_.end()) {
      Say(source, "Unused USE statement for module '%s'"_warn_en_US,
          moduleSymbol->name());
    }
  }

  importedSymbols_.clear();
  usedSymbols_.clear();
  wholeModuleImports_.clear();
  usedModules_.clear();
}

} // namespace Fortran::tidy::readability
