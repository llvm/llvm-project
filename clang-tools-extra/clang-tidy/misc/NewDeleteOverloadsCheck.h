//===--- NewDeleteOverloadsCheck.h - clang-tidy----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_NEWDELETEOVERLOADS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_NEWDELETEOVERLOADS_H

#include "../ClangTidyCheck.h"
#include "llvm/ADT/SmallVector.h"
#include <map>

namespace clang::tidy::misc {

class NewDeleteOverloadsCheck : public ClangTidyCheck {
  std::map<const clang::CXXRecordDecl *,
           llvm::SmallVector<const clang::FunctionDecl *, 4>>
      Overloads;

public:
  NewDeleteOverloadsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void onEndOfTranslationUnit() override;
};

} // namespace clang::tidy::misc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_NEWDELETEOVERLOADS_H
