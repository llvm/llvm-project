//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_OPERATORSREPRESENTATIONCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_OPERATORSREPRESENTATIONCHECK_H

#include "../ClangTidyCheck.h"
#include <vector>

namespace clang::tidy::readability {

/// Enforces consistent token representation for invoked binary, unary
/// and overloaded operators in C++ code.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability/operators-representation.html
class OperatorsRepresentationCheck : public ClangTidyCheck {
public:
  OperatorsRepresentationCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;
  std::optional<TraversalKind> getCheckTraversalKind() const override;

private:
  void registerBinaryOperatorMatcher(ast_matchers::MatchFinder *Finder);
  void registerUnaryOperatorMatcher(ast_matchers::MatchFinder *Finder);
  void registerOverloadedOperatorMatcher(ast_matchers::MatchFinder *Finder);

  std::vector<llvm::StringRef> BinaryOperators;
  std::vector<llvm::StringRef> OverloadedOperators;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_OPERATORSREPRESENTATIONCHECK_H
