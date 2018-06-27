//===---------- ExprMutationAnalyzer.h - clang-tidy -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_EXPRMUTATIONANALYZER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_EXPRMUTATIONANALYZER_H

#include <type_traits>

#include "clang/AST/AST.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {
namespace tidy {
namespace utils {

/// Analyzes whether any mutative operations are applied to an expression within
/// a given statement.
class ExprMutationAnalyzer {
public:
  ExprMutationAnalyzer(const Stmt *Stm, ASTContext *Context)
      : Stm(Stm), Context(Context) {}

  bool isMutated(const Decl *Dec) { return findDeclMutation(Dec) != nullptr; }
  bool isMutated(const Expr *Exp) { return findMutation(Exp) != nullptr; }
  const Stmt *findMutation(const Expr *Exp);

private:
  bool isUnevaluated(const Expr *Exp);

  const Stmt *findExprMutation(ArrayRef<ast_matchers::BoundNodes> Matches);
  const Stmt *findDeclMutation(ArrayRef<ast_matchers::BoundNodes> Matches);
  const Stmt *findDeclMutation(const Decl *Dec);

  const Stmt *findDirectMutation(const Expr *Exp);
  const Stmt *findMemberMutation(const Expr *Exp);
  const Stmt *findArrayElementMutation(const Expr *Exp);
  const Stmt *findCastMutation(const Expr *Exp);
  const Stmt *findRangeLoopMutation(const Expr *Exp);
  const Stmt *findReferenceMutation(const Expr *Exp);

  const Stmt *const Stm;
  ASTContext *const Context;
  llvm::DenseMap<const Expr *, const Stmt *> Results;
};

} // namespace utils
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_EXPRMUTATIONANALYZER_H
