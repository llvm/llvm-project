//===--- UnusedUSECheck.h - flang-tidy --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_TOOLS_FLANG_TIDY_READABILITY_UNUSEUSECHECK_H
#define LLVM_FLANG_TOOLS_FLANG_TIDY_READABILITY_UNUSEUSECHECK_H

#include "../FlangTidyCheck.h"
#include <map>
#include <set>

namespace Fortran::tidy::readability {

class UnusedUSECheck : public FlangTidyCheck {
public:
  UnusedUSECheck(llvm::StringRef name, FlangTidyContext *context);

  void Enter(const parser::UseStmt &) override;
  void Enter(const parser::Name &) override;
  void Leave(const parser::ProgramUnit &) override;

private:
  // Track imported symbols: symbol -> source location
  std::map<const semantics::Symbol *, parser::CharBlock> importedSymbols_;

  // Track used symbols
  std::set<const semantics::Symbol *> usedSymbols_;

  // Track whole module imports: module symbol -> source location
  std::map<const semantics::Symbol *, parser::CharBlock> wholeModuleImports_;

  // Track used modules
  std::set<const semantics::Symbol *> usedModules_;
};

} // namespace Fortran::tidy::readability

#endif // LLVM_FLANG_TOOLS_FLANG_TIDY_READABILITY_UNUSEUSECHECK_H
