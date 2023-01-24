//===--- IntegerTypesCheck.h - clang-tidy -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_INTEGERTYPESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_INTEGERTYPESCHECK_H

#include "../ClangTidyCheck.h"

#include <memory>

namespace clang {

class IdentifierTable;

namespace tidy::google::runtime {

/// Finds uses of `short`, `long` and `long long` and suggest replacing them
/// with `u?intXX(_t)?`.
///
/// Corresponding cpplint.py check: 'runtime/int'.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/google/runtime-int.html
class IntegerTypesCheck : public ClangTidyCheck {
public:
  IntegerTypesCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus && !LangOpts.ObjC;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Options) override;

private:
  const StringRef UnsignedTypePrefix;
  const StringRef SignedTypePrefix;
  const StringRef TypeSuffix;

  std::unique_ptr<IdentifierTable> IdentTable;
};

} // namespace tidy::google::runtime
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_INTEGERTYPESCHECK_H
