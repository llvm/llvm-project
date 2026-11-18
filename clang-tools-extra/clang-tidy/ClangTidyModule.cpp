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

namespace clang::tidy {

ClangTidyCheckFactories::FactoryEntry::FactoryEntry(
    CheckFactoryFunction Function)
    : Function(Function) {}

ClangTidyCheckFactories::FactoryEntry::FactoryEntry(CheckFactory Factory)
    : Factory(std::make_unique<CheckFactory>(std::move(Factory))) {}

ClangTidyCheckFactories::FactoryEntry::FactoryEntry(const FactoryEntry &Other)
    : Function(Other.Function) {
  if (Other.Factory)
    Factory = std::make_unique<CheckFactory>(*Other.Factory);
}

ClangTidyCheckFactories::FactoryEntry &
ClangTidyCheckFactories::FactoryEntry::operator=(const FactoryEntry &Other) {
  if (this == &Other)
    return *this;
  Function = Other.Function;
  Factory =
      Other.Factory ? std::make_unique<CheckFactory>(*Other.Factory) : nullptr;
  return *this;
}

std::unique_ptr<ClangTidyCheck>
ClangTidyCheckFactories::FactoryEntry::operator()(
    StringRef Name, ClangTidyContext *Context) const {
  if (Function)
    return Function(Name, Context);
  return (*Factory)(Name, Context);
}

void ClangTidyCheckFactories::registerCheckFactory(StringRef Name,
                                                   CheckFactory Factory) {
  Factories.insert_or_assign(Name, FactoryEntry(std::move(Factory)));
}

void ClangTidyCheckFactories::registerCheckFactory(
    StringRef Name, const FactoryEntry &Factory) {
  Factories.insert_or_assign(Name, Factory);
}

void ClangTidyCheckFactories::registerCheckFunction(
    StringRef Name, CheckFactoryFunction Function) {
  Factories.insert_or_assign(Name, FactoryEntry(Function));
}

std::vector<std::unique_ptr<ClangTidyCheck>>
ClangTidyCheckFactories::createChecks(ClangTidyContext *Context) const {
  std::vector<std::unique_ptr<ClangTidyCheck>> Checks;
  for (const auto &[CheckName, Factory] : Factories)
    if (Context->isCheckEnabled(CheckName))
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
    std::unique_ptr<ClangTidyCheck> Check = Factory(CheckName, Context);
    if (Check->isLanguageVersionSupported(LO))
      Checks.push_back(std::move(Check));
  }
  return Checks;
}

ClangTidyOptions ClangTidyModule::getModuleOptions() { return {}; }

} // namespace clang::tidy
