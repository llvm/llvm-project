//===--- tools/extra/clang-tidy/ClangTidyModule.cpp - Clang tidy tool -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  \file Implements classes required to build clang-tidy modules.
///
//===----------------------------------------------------------------------===//

#include "ClangTidyModule.h"

namespace clang {
namespace tidy {

void ClangTidyCheckFactories::registerCheckFactory(StringRef Name,
                                                   CheckFactory Factory) {
  Factories[Name] = std::move(Factory);
}

void ClangTidyCheckFactories::createChecks(
    ClangTidyContext *Context,
    std::vector<std::unique_ptr<ClangTidyCheck>> &Checks) {
  GlobList &Filter = Context->getChecksFilter();
  for (const auto &Factory : Factories) {
    if (Filter.contains(Factory.first))
      Checks.emplace_back(Factory.second(Factory.first, Context));
  }
}

ClangTidyOptions ClangTidyModule::getModuleOptions() {
  return ClangTidyOptions();
}

} // namespace tidy
} // namespace clang
