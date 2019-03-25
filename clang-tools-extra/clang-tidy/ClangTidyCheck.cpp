//===--- ClangTidyCheck.cpp - clang-tidy ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangTidyCheck.h"

namespace clang {
namespace tidy {

ClangTidyCheck::ClangTidyCheck(StringRef CheckName, ClangTidyContext *Context)
    : CheckName(CheckName), Context(Context),
      Options(CheckName, Context->getOptions().CheckOptions) {
  assert(Context != nullptr);
  assert(!CheckName.empty());
}

DiagnosticBuilder ClangTidyCheck::diag(SourceLocation Loc, StringRef Message,
                                       DiagnosticIDs::Level Level) {
  return Context->diag(CheckName, Loc, Message, Level);
}

void ClangTidyCheck::run(const ast_matchers::MatchFinder::MatchResult &Result) {
  // For historical reasons, checks don't implement the MatchFinder run()
  // callback directly. We keep the run()/check() distinction to avoid interface
  // churn, and to allow us to add cross-cutting logic in the future.
  check(Result);
}

ClangTidyCheck::OptionsView::OptionsView(StringRef CheckName,
                         const ClangTidyOptions::OptionMap &CheckOptions)
    : NamePrefix(CheckName.str() + "."), CheckOptions(CheckOptions) {}

std::string ClangTidyCheck::OptionsView::get(StringRef LocalName,
                                             StringRef Default) const {
  const auto &Iter = CheckOptions.find(NamePrefix + LocalName.str());
  if (Iter != CheckOptions.end())
    return Iter->second;
  return Default;
}

std::string
ClangTidyCheck::OptionsView::getLocalOrGlobal(StringRef LocalName,
                                              StringRef Default) const {
  auto Iter = CheckOptions.find(NamePrefix + LocalName.str());
  if (Iter != CheckOptions.end())
    return Iter->second;
  // Fallback to global setting, if present.
  Iter = CheckOptions.find(LocalName.str());
  if (Iter != CheckOptions.end())
    return Iter->second;
  return Default;
}

void ClangTidyCheck::OptionsView::store(ClangTidyOptions::OptionMap &Options,
                                        StringRef LocalName,
                                        StringRef Value) const {
  Options[NamePrefix + LocalName.str()] = Value;
}

void ClangTidyCheck::OptionsView::store(ClangTidyOptions::OptionMap &Options,
                                        StringRef LocalName,
                                        int64_t Value) const {
  store(Options, LocalName, llvm::itostr(Value));
}

} // namespace tidy
} // namespace clang
