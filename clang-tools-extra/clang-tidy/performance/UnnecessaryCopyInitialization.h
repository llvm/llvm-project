//===--- UnnecessaryCopyInitialization.h - clang-tidy------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_UNNECESSARY_COPY_INITIALIZATION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_UNNECESSARY_COPY_INITIALIZATION_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace performance {

// The check detects local variable declarations that are copy initialized with
// the const reference of a function call or the const reference of a method
// call whose object is guaranteed to outlive the variable's scope and suggests
// to use a const reference.
//
// The check currently only understands a subset of variables that are
// guaranteed to outlive the const reference returned, namely: const variables,
// const references, and const pointers to const.
class UnnecessaryCopyInitialization : public ClangTidyCheck {
public:
  UnnecessaryCopyInitialization(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  void handleCopyFromMethodReturn(const VarDecl &Var, const Stmt &BlockStmt,
                                  bool IssueFix, const VarDecl *ObjectArg,
                                  ASTContext &Context);
  void handleCopyFromLocalVar(const VarDecl &NewVar, const VarDecl &OldVar,
                              const Stmt &BlockStmt, bool IssueFix,
                              ASTContext &Context);
  const std::vector<std::string> AllowedTypes;
};

} // namespace performance
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_UNNECESSARY_COPY_INITIALIZATION_H
