//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_NAMEDPARAMETERCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_NAMEDPARAMETERCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// Find functions with unnamed arguments.
///
/// The check implements the following rule originating in the Google C++ Style
/// Guide:
///
/// https://google.github.io/styleguide/cppguide.html#Function_Declarations_and_Definitions
///
/// All parameters should be named, with identical names in the declaration and
/// implementation.
///
/// Corresponding cpplint.py check name: 'readability/function'.
class NamedParameterCheck : public ClangTidyCheck {
public:
  NamedParameterCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  const bool InsertPlainNamesInForwardDecls;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_NAMEDPARAMETERCHECK_H
