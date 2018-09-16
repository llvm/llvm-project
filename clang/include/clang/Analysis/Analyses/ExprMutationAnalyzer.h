//===---------- ExprMutationAnalyzer.h ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_EXPRMUTATIONANALYZER_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_EXPRMUTATIONANALYZER_H

#include <type_traits>

#include "clang/AST/AST.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {

class FunctionParmMutationAnalyzer;

/// Analyzes whether any mutative operations are applied to an expression within
/// a given statement.
class ExprMutationAnalyzer {
public:
  ExprMutationAnalyzer(const Stmt &Stm, ASTContext &Context)
      : Stm(Stm), Context(Context) {}

  bool isMutated(const Decl *Dec) { return findDeclMutation(Dec) != nullptr; }
  bool isMutated(const Expr *Exp) { return findMutation(Exp) != nullptr; }
  const Stmt *findMutation(const Expr *Exp);
  const Stmt *findDeclMutation(const Decl *Dec);

private:
  bool isUnevaluated(const Expr *Exp);

  const Stmt *findExprMutation(ArrayRef<ast_matchers::BoundNodes> Matches);
  const Stmt *findDeclMutation(ArrayRef<ast_matchers::BoundNodes> Matches);

  const Stmt *findDirectMutation(const Expr *Exp);
  const Stmt *findMemberMutation(const Expr *Exp);
  const Stmt *findArrayElementMutation(const Expr *Exp);
  const Stmt *findCastMutation(const Expr *Exp);
  const Stmt *findRangeLoopMutation(const Expr *Exp);
  const Stmt *findReferenceMutation(const Expr *Exp);
  const Stmt *findFunctionArgMutation(const Expr *Exp);

  const Stmt &Stm;
  ASTContext &Context;
  llvm::DenseMap<const FunctionDecl *,
                 std::unique_ptr<FunctionParmMutationAnalyzer>>
      FuncParmAnalyzer;
  llvm::DenseMap<const Expr *, const Stmt *> Results;
};

// A convenient wrapper around ExprMutationAnalyzer for analyzing function
// params.
class FunctionParmMutationAnalyzer {
public:
  FunctionParmMutationAnalyzer(const FunctionDecl &Func, ASTContext &Context);

  bool isMutated(const ParmVarDecl *Parm) {
    return findMutation(Parm) != nullptr;
  }
  const Stmt *findMutation(const ParmVarDecl *Parm);

private:
  ExprMutationAnalyzer BodyAnalyzer;
  llvm::DenseMap<const ParmVarDecl *, const Stmt *> Results;
};

} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_EXPRMUTATIONANALYZER_H
