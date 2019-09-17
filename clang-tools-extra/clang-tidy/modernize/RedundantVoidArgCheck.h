//===--- RedundantVoidArgCheck.h - clang-tidy --------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_REDUNDANT_VOID_ARG_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_REDUNDANT_VOID_ARG_CHECK_H

#include "../ClangTidyCheck.h"
#include "clang/Lex/Token.h"

#include <string>

namespace clang {
namespace tidy {
namespace modernize {

/// Find and remove redundant void argument lists.
///
/// Examples:
///   `int f(void);`                    becomes `int f();`
///   `int (*f(void))(void);`           becomes `int (*f())();`
///   `typedef int (*f_t(void))(void);` becomes `typedef int (*f_t())();`
///   `void (C::*p)(void);`             becomes `void (C::*p)();`
///   `C::C(void) {}`                   becomes `C::C() {}`
///   `C::~C(void) {}`                  becomes `C::~C() {}`
///
class RedundantVoidArgCheck : public ClangTidyCheck {
public:
  RedundantVoidArgCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;

  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void processFunctionDecl(const ast_matchers::MatchFinder::MatchResult &Result,
                           const FunctionDecl *Function);

  void
  processTypedefNameDecl(const ast_matchers::MatchFinder::MatchResult &Result,
                         const TypedefNameDecl *Typedef);

  void processFieldDecl(const ast_matchers::MatchFinder::MatchResult &Result,
                        const FieldDecl *Member);

  void processVarDecl(const ast_matchers::MatchFinder::MatchResult &Result,
                      const VarDecl *Var);

  void
  processNamedCastExpr(const ast_matchers::MatchFinder::MatchResult &Result,
                       const CXXNamedCastExpr *NamedCast);

  void
  processExplicitCastExpr(const ast_matchers::MatchFinder::MatchResult &Result,
                          const ExplicitCastExpr *ExplicitCast);

  void processLambdaExpr(const ast_matchers::MatchFinder::MatchResult &Result,
                         const LambdaExpr *Lambda);

  void
  removeVoidArgumentTokens(const ast_matchers::MatchFinder::MatchResult &Result,
                           SourceRange Range, StringRef GrammarLocation);

  void removeVoidToken(Token VoidToken, StringRef Diagnostic);
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_REDUNDANT_VOID_ARG_CHECK_H
