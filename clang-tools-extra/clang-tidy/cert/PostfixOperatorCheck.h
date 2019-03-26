//===--- PostfixOperatorCheck.h - clang-tidy---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_POSTFIX_OPERATOR_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_POSTFIX_OPERATOR_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace cert {

/// Checks if the overloaded postfix ++ and -- operator return a constant
/// object.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cert-postfix-operator.html
class PostfixOperatorCheck : public ClangTidyCheck {
public:
  PostfixOperatorCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace cert
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_POSTFIX_OPERATOR_H
