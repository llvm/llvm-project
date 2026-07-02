//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
///  \file Implements classes required to build clang-tidy modules.
///
//===----------------------------------------------------------------------===//

#include "ClangTidyModule.h"
#include "ClangTidyCheck.h"
#include "aliases/ClangTidyAliases.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace clang::tidy {

/// Returns true if CheckName is an alias whose canonical check is also enabled.
static bool isRedundantAlias(StringRef CheckName, ClangTidyContext *Context) {
  const StringRef Canonical = ClangTidyAliases::getCanonicalForAlias(CheckName);
  return !Canonical.empty() && Context->isCheckEnabled(Canonical);
}

void ClangTidyCheckFactories::registerCheckFactory(StringRef Name,
                                                   CheckFactory Factory) {
  Factories.insert_or_assign(Name, std::move(Factory));
}

std::vector<std::unique_ptr<ClangTidyCheck>>
ClangTidyCheckFactories::createChecks(ClangTidyContext *Context) const {
  std::vector<std::unique_ptr<ClangTidyCheck>> Checks;
  for (const auto &[CheckName, Factory] : Factories)
    if (Context->isCheckEnabled(CheckName) &&
        !isRedundantAlias(CheckName, Context))
      Checks.emplace_back(Factory(CheckName, Context));
  return Checks;
}

std::vector<std::unique_ptr<ClangTidyCheck>>
ClangTidyCheckFactories::createChecksForLanguage(
    ClangTidyContext *Context) const {
  std::vector<std::unique_ptr<ClangTidyCheck>> Checks;
  const LangOptions &LO = Context->getLangOpts();
  for (const auto &[CheckName, Factory] : Factories) {
    if (!Context->isCheckEnabled(CheckName))
      continue;
    if (isRedundantAlias(CheckName, Context))
      continue;
    std::unique_ptr<ClangTidyCheck> Check = Factory(CheckName, Context);
    if (Check->isLanguageVersionSupported(LO))
      Checks.push_back(std::move(Check));
  }
  return Checks;
}

ClangTidyOptions ClangTidyModule::getModuleOptions() { return {}; }

} // namespace clang::tidy
