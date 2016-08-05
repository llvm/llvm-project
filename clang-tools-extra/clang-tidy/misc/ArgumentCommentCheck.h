//===--- ArgumentCommentCheck.h - clang-tidy --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_ARGUMENTCOMMENTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_ARGUMENTCOMMENTCHECK_H

#include "../ClangTidy.h"
#include "llvm/Support/Regex.h"

namespace clang {
namespace tidy {
namespace misc {

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
///   // warning: argument name 'bar' in comment does not match parameter name 'foo'
/// \endcode
///
/// The check tries to detect typos and suggest automated fixes for them.
class ArgumentCommentCheck : public ClangTidyCheck {
public:
  ArgumentCommentCheck(StringRef Name, ClangTidyContext *Context);

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap& Opts) override;

private:
  const bool StrictMode;
  llvm::Regex IdentRE;

  bool isLikelyTypo(llvm::ArrayRef<ParmVarDecl *> Params, StringRef ArgName,
                    unsigned ArgIndex);
  void checkCallArgs(ASTContext *Ctx, const FunctionDecl *Callee,
                     SourceLocation ArgBeginLoc,
                     llvm::ArrayRef<const Expr *> Args);
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_ARGUMENTCOMMENTCHECK_H
