//===-- LoopConvert/StmtAncestor.h - AST property visitors ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the declarations of several RecursiveASTVisitors
/// used to build and check data structures used in loop migration.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_STMT_ANCESTOR_H
#define LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_STMT_ANCESTOR_H

#include "clang/AST/RecursiveASTVisitor.h"

/// A map used to walk the AST in reverse: maps child Stmt to parent Stmt.
typedef llvm::DenseMap<const clang::Stmt*, const clang::Stmt*> StmtParentMap;

/// A map used to walk the AST in reverse:
///  maps VarDecl to the to parent DeclStmt.
typedef
llvm::DenseMap<const clang::VarDecl*, const clang::DeclStmt*> DeclParentMap;

/// A map used to track which variables have been removed by a refactoring pass.
/// It maps the parent ForStmt to the removed index variable's VarDecl.
typedef
llvm::DenseMap<const clang::ForStmt*, const clang::VarDecl*> ReplacedVarsMap;

/// A map used to remember the variable names generated in a Stmt
typedef llvm::DenseMap<const clang::Stmt*, std::string> StmtGeneratedVarNameMap;

/// A vector used to store the AST subtrees of an Expr.
typedef llvm::SmallVector<const clang::Expr*, 16> ComponentVector;

/// \brief Class used build the reverse AST properties needed to detect
/// name conflicts and free variables.
class StmtAncestorASTVisitor :
  public clang::RecursiveASTVisitor<StmtAncestorASTVisitor> {
public:
  StmtAncestorASTVisitor() {
    StmtStack.push_back(NULL);
  }

  /// \brief Run the analysis on the TranslationUnitDecl.
  ///
  /// In case we're running this analysis multiple times, don't repeat the work.
  void gatherAncestors(const clang::TranslationUnitDecl *T) {
    if (StmtAncestors.empty())
      TraverseDecl(const_cast<clang::TranslationUnitDecl*>(T));
  }

  /// Accessor for StmtAncestors.
  const StmtParentMap &getStmtToParentStmtMap() {
    return StmtAncestors;
  }

  /// Accessor for DeclParents.
  const DeclParentMap &getDeclToParentStmtMap() {
    return DeclParents;
  }

  friend class clang::RecursiveASTVisitor<StmtAncestorASTVisitor>;

private:
  StmtParentMap StmtAncestors;
  DeclParentMap DeclParents;
  llvm::SmallVector<const clang::Stmt*, 16> StmtStack;

  bool TraverseStmt(clang::Stmt *Statement);
  bool VisitDeclStmt(clang::DeclStmt *Statement);
};

/// Class used to find the variables and member expressions on which an
/// arbitrary expression depends.
class ComponentFinderASTVisitor :
  public clang::RecursiveASTVisitor<ComponentFinderASTVisitor> {
public:
  ComponentFinderASTVisitor() { }

  /// Find the components of an expression and place them in a ComponentVector.
  void findExprComponents(const clang::Expr *SourceExpr) {
    clang::Expr *E = const_cast<clang::Expr *>(SourceExpr);
    TraverseStmt(E);
  }

  /// Accessor for Components.
  const ComponentVector &getComponents() {
    return Components;
  }

  friend class clang::RecursiveASTVisitor<ComponentFinderASTVisitor>;

private:
  ComponentVector Components;

  bool VisitDeclRefExpr(clang::DeclRefExpr *E);
  bool VisitMemberExpr(clang::MemberExpr *Member);
};

/// Class used to determine if an expression is dependent on a variable declared
/// inside of the loop where it would be used.
class DependencyFinderASTVisitor :
  public clang::RecursiveASTVisitor<DependencyFinderASTVisitor> {
public:
  DependencyFinderASTVisitor(const StmtParentMap *StmtParents,
                             const DeclParentMap *DeclParents,
                             const ReplacedVarsMap *ReplacedVars,
                             const clang::Stmt *ContainingStmt) :
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
  bool dependsOnInsideVariable(const clang::Stmt *Body) {
    DependsOnInsideVariable = false;
    TraverseStmt(const_cast<clang::Stmt *>(Body));
    return DependsOnInsideVariable;
  }

  friend class clang::RecursiveASTVisitor<DependencyFinderASTVisitor>;

private:
  const StmtParentMap *StmtParents;
  const DeclParentMap *DeclParents;
  const clang::Stmt *ContainingStmt;
  const ReplacedVarsMap *ReplacedVars;
  bool DependsOnInsideVariable;

  bool VisitVarDecl(clang::VarDecl *V);
  bool VisitDeclRefExpr(clang::DeclRefExpr *D);
};

/// Class used to determine if any declarations used in a Stmt would conflict
/// with a particular identifier. This search includes the names that don't
/// actually appear in the AST (i.e. created by a refactoring tool) by including
/// a map from Stmts to generated names associated with those stmts.
class DeclFinderASTVisitor :
  public clang::RecursiveASTVisitor<DeclFinderASTVisitor> {
public:
  DeclFinderASTVisitor(const std::string &Name,
                       const StmtGeneratedVarNameMap *GeneratedDecls) :
    Name(Name), GeneratedDecls(GeneratedDecls), Found(false) { }

  /// Attempts to find any usages of variables name Name in Body, returning
  /// true when it is used in Body. This includes the generated loop variables
  /// of ForStmts which have already been transformed.
  bool findUsages(const clang::Stmt *Body) {
    Found = false;
    TraverseStmt(const_cast<clang::Stmt *>(Body));
    return Found;
  }

  friend class clang::RecursiveASTVisitor<DeclFinderASTVisitor>;

private:
  std::string Name;
  /// GeneratedDecls keeps track of ForStmts which have been tranformed, mapping
  /// each modified ForStmt to the variable generated in the loop.
  const StmtGeneratedVarNameMap *GeneratedDecls;
  bool Found;

  bool VisitForStmt(clang::ForStmt *F);
  bool VisitNamedDecl(clang::NamedDecl *D);
  bool VisitDeclRefExpr(clang::DeclRefExpr *D);
  bool VisitTypeLoc(clang::TypeLoc TL);
};

#endif // LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_STMT_ANCESTOR_H
