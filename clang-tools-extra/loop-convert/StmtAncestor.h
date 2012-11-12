//===-- loop-convert/StmtAncestor.h - AST property visitors -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of several RecursiveASTVisitors used to
// build and check data structures used in loop migration.
//
//===----------------------------------------------------------------------===//
#ifndef _LLVM_TOOLS_CLANG_TOOLS_EXTRA_LOOP_CONVERT_STMT_ANCESTOR_H_
#define _LLVM_TOOLS_CLANG_TOOLS_EXTRA_LOOP_CONVERT_STMT_ANCESTOR_H_
#include "clang/AST/RecursiveASTVisitor.h"

namespace clang {
namespace loop_migrate {

/// A map used to walk the AST in reverse: maps child Stmt to parent Stmt.
typedef llvm::DenseMap<const Stmt*, const Stmt*> StmtParentMap;

/// A map used to walk the AST in reverse:
///  maps VarDecl to the to parent DeclStmt.
typedef llvm::DenseMap<const VarDecl*, const DeclStmt*> DeclParentMap;

/// A map used to track which variables have been removed by a refactoring pass.
/// It maps the parent ForStmt to the removed index variable's VarDecl.
typedef llvm::DenseMap<const ForStmt*, const VarDecl *> ReplacedVarsMap;

/// A map used to remember the variable names generated in a Stmt
typedef llvm::DenseMap<const Stmt*, std::string> StmtGeneratedVarNameMap;

/// A vector used to store the AST subtrees of an Expr.
typedef llvm::SmallVector<const Expr *, 16> ComponentVector;

/// \brief Class used build the reverse AST properties needed to detect
/// name conflicts and free variables.
class StmtAncestorASTVisitor :
  public RecursiveASTVisitor<StmtAncestorASTVisitor> {
 public:
  StmtAncestorASTVisitor() {
    StmtStack.push_back(NULL);
  }

  /// \brief Run the analysis on the TranslationUnitDecl.
  ///
  /// In case we're running this analysis multiple times, don't repeat the work.
  void gatherAncestors(const TranslationUnitDecl *TUD) {
    if (StmtAncestors.empty())
      TraverseDecl(const_cast<TranslationUnitDecl *>(TUD));
  }

  /// Accessor for StmtAncestors.
  const StmtParentMap &getStmtToParentStmtMap() {
    return StmtAncestors;
  }

  /// Accessor for DeclParents.
  const DeclParentMap &getDeclToParentStmtMap() {
    return DeclParents;
  }

  friend class RecursiveASTVisitor<StmtAncestorASTVisitor>;

 private:
  StmtParentMap StmtAncestors;
  DeclParentMap DeclParents;
  llvm::SmallVector<const Stmt *, 16> StmtStack;

  bool TraverseStmt(Stmt *Statement);
  bool VisitDeclStmt(DeclStmt *Statement);
};

/// Class used to find the variables and member expressions on which an
/// arbitrary expression depends.
class ComponentFinderASTVisitor :
  public RecursiveASTVisitor<ComponentFinderASTVisitor> {
 public:
  ComponentFinderASTVisitor() { }

  /// Find the components of an expression and place them in a ComponentVector.
  void findExprComponents(const Expr *SourceExpr) {
    Expr *E = const_cast<Expr *>(SourceExpr);
    RecursiveASTVisitor<ComponentFinderASTVisitor>::TraverseStmt(E);
  }

  /// Accessor for Components.
  const ComponentVector &getComponents() {
    return Components;
  }

  friend class RecursiveASTVisitor<ComponentFinderASTVisitor>;

 private:
  ComponentVector Components;

  bool VisitDeclRefExpr(DeclRefExpr *E);
  bool VisitMemberExpr(MemberExpr *Member);
};

/// Class used to determine if an expression is dependent on a variable declared
/// inside of the loop where it would be used.
class DependencyFinderASTVisitor :
  public RecursiveASTVisitor<DependencyFinderASTVisitor> {
 public:
  DependencyFinderASTVisitor(const StmtParentMap *StmtParents,
                             const DeclParentMap *DeclParents,
                             const ReplacedVarsMap *ReplacedVars,
                             const Stmt *ContainingStmt) :
    StmtParents(StmtParents), DeclParents(DeclParents),
    ContainingStmt(ContainingStmt), ReplacedVars(ReplacedVars) { }

  /// \brief Run the analysis on Body, and return true iff the expression
  /// depends on some variable declared within ContainingStmt.
  ///
  /// This is intended to protect against hoisting the container expression
  /// outside of an inner context if part of that expression is declared in that
  /// inner context.
  ///
  /// For example,
  /// \code
  ///   const int N = 10, M = 20;
  ///   int arr[N][M];
  ///   int getRow();
  ///
  ///   for (int i = 0; i < M; ++i) {
  ///     int k = getRow();
  ///     printf("%d:", arr[k][i]);
  ///   }
  /// \endcode
  /// At first glance, this loop looks like it could be changed to
  /// \code
  ///   for (int elem : arr[k]) {
  ///     int k = getIndex();
  ///     printf("%d:", elem);
  ///   }
  /// \endcode
  /// But this is malformed, since `k` is used before it is defined!
  ///
  /// In order to avoid this, this class looks at the container expression
  /// `arr[k]` and decides whether or not it contains a sub-expression declared
  /// within the the loop body.
  bool dependsOnInsideVariable(const Stmt *Body) {
    DependsOnInsideVariable = false;
    TraverseStmt(const_cast<Stmt *>(Body));
    return DependsOnInsideVariable;
  }

  friend class RecursiveASTVisitor<DependencyFinderASTVisitor>;

 private:
  const StmtParentMap *StmtParents;
  const DeclParentMap *DeclParents;
  const Stmt *ContainingStmt;
  const ReplacedVarsMap *ReplacedVars;
  bool DependsOnInsideVariable;

  bool VisitVarDecl(VarDecl *VD);
  bool VisitDeclRefExpr(DeclRefExpr *DRE);
};

/// Class used to determine if any declarations used in a Stmt would conflict
/// with a particular identifier. This search includes the names that don't
/// actually appear in the AST (i.e. created by a refactoring tool) by including
/// a map from Stmts to generated names associated with those stmts.
class DeclFinderASTVisitor : public RecursiveASTVisitor<DeclFinderASTVisitor> {
 public:
  DeclFinderASTVisitor(const std::string &Name,
                       const StmtGeneratedVarNameMap *GeneratedDecls) :
    Name(Name), GeneratedDecls(GeneratedDecls), Found(false) { }

  /// Attempts to find any usages of variables name Name in Body, returning
  /// true when it is used in Body. This includes the generated loop variables
  /// of ForStmts which have already been transformed.
  bool findUsages(const Stmt *Body) {
    Found = false;
    TraverseStmt(const_cast<Stmt *>(Body));
    return Found;
  }

  friend class RecursiveASTVisitor<DeclFinderASTVisitor>;

 private:
  std::string Name;
  /// GeneratedDecls keeps track of ForStmts which have been tranformed, mapping
  /// each modified ForStmt to the variable generated in the loop.
  const StmtGeneratedVarNameMap *GeneratedDecls;
  bool Found;

  bool VisitForStmt(ForStmt *FS);
  bool VisitNamedDecl(NamedDecl *ND);
  bool VisitDeclRefExpr(DeclRefExpr *DRE);
};

} // namespace for_migrate
} // namespace clang
#endif // _LLVM_TOOLS_CLANG_TOOLS_EXTRA_LOOP_CONVERT_STMT_ANCESTOR_H_
