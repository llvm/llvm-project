//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USENULLPTRCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USENULLPTRCHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/IncludeInserter.h"
#include "llvm/ADT/SmallVectorWithFlags.h"

namespace clang::tidy::modernize {

class UseNullptrCheck : public ClangTidyCheck {
public:
  UseNullptrCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11 || LangOpts.C23;
  }
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const StringRef NullMacrosStr;
  llvm::SmallVectorWithFlags<StringRef, 1> NullMacros;
  std::vector<StringRef> IgnoredTypes;
  bool getUseNullptrT() const { return NullMacros.getFlag(0); }
  utils::IncludeInserter IncludeInserter;

public:
  UseNullptrCheck(const UseNullptrCheck &) = default;
  UseNullptrCheck(UseNullptrCheck &&) = default;
  UseNullptrCheck &operator=(const UseNullptrCheck &) = delete;
  UseNullptrCheck &operator=(UseNullptrCheck &&) = delete;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USENULLPTRCHECK_H
