//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REDUNDANTSMARTPTRGETCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REDUNDANTSMARTPTRGETCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// Find and remove redundant calls to smart pointer's `.get()` method.
///
/// Examples:
///
/// \code
///   ptr.get()->Foo()  ==>  ptr->Foo()
///   *ptr.get()  ==>  *ptr
///   *ptr->get()  ==>  **ptr
/// \endcode
class RedundantSmartptrGetCheck : public ClangTidyCheck {
public:
  RedundantSmartptrGetCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        IgnoreMacros(Options.get("IgnoreMacros", true)) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  const bool IgnoreMacros;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REDUNDANTSMARTPTRGETCHECK_H
