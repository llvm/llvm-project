//===--- MoveSharedPtrCheck.h - clang-tidy ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_MOVESHAREDPTRCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_MOVESHAREDPTRCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::performance {

/// adds std::move() to std::shared_ptr.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/performance/move-shared-ptr.html
class MoveSharedPtrCheck : public ClangTidyCheck {
public:
  MoveSharedPtrCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  
private:
  const Expr* getLastVarUsage(const VarDecl& Var, const Decl& Func, ASTContext &Context);

  llvm::DenseMap<const VarDecl *, SourceLocation> last_usage_;
};

} // namespace clang::tidy::performance

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_MOVESHAREDPTRCHECK_H
