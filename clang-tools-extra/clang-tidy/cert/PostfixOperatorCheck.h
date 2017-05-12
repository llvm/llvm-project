//===--- PostfixOperatorCheck.h - clang-tidy---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_POSTFIX_OPERATOR_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_POSTFIX_OPERATOR_H

#include "../ClangTidy.h"

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
