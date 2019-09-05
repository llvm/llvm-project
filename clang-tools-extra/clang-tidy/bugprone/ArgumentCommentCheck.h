//===--- ArgumentCommentCheck.h - clang-tidy --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_ARGUMENTCOMMENTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_ARGUMENTCOMMENTCHECK_H

#include "../ClangTidyCheck.h"
#include "llvm/Support/Regex.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// Checks that argument comments match parameter names.
///
/// The check understands argument comments in the form `/*parameter_name=*/`
/// that are placed right before the argument.
///
/// \code
///   void f(bool foo);
///
///   ...
///   f(/*bar=*/true);
///   // warning: argument name 'bar' in comment does not match parameter name
///   'foo'
/// \endcode
///
/// The check tries to detect typos and suggest automated fixes for them.
class ArgumentCommentCheck : public ClangTidyCheck {
public:
  ArgumentCommentCheck(StringRef Name, ClangTidyContext *Context);

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  const unsigned StrictMode : 1;
  const unsigned IgnoreSingleArgument : 1;
  const unsigned CommentBoolLiterals : 1;
  const unsigned CommentIntegerLiterals : 1;
  const unsigned CommentFloatLiterals : 1;
  const unsigned CommentStringLiterals : 1;
  const unsigned CommentUserDefinedLiterals : 1;
  const unsigned CommentCharacterLiterals : 1;
  const unsigned CommentNullPtrs : 1;
  llvm::Regex IdentRE;

  void checkCallArgs(ASTContext *Ctx, const FunctionDecl *Callee,
                     SourceLocation ArgBeginLoc,
                     llvm::ArrayRef<const Expr *> Args);

  bool shouldAddComment(const Expr *Arg) const;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_ARGUMENTCOMMENTCHECK_H
