//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_AMBIGUOUSSMARTPTRRESETCALLCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_AMBIGUOUSSMARTPTRRESETCALLCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// Finds potentially erroneous calls to 'reset' method on smart pointers when
/// the pointee type also has a 'reset' method
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability/ambiguous-smartptr-reset-call.html
class AmbiguousSmartptrResetCallCheck : public ClangTidyCheck {
public:
  AmbiguousSmartptrResetCallCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }

private:
  const std::vector<StringRef> SmartPointers;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_AMBIGUOUSSMARTPTRRESETCALLCHECK_H
