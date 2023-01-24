//===--- LoopConvertUtils.h - clang-tidy ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_LOOP_CONVERT_UTILS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_LOOP_CONVERT_UTILS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <algorithm>
#include <memory>
#include <string>
#include <utility>

namespace clang::tidy::modernize {

enum LoopFixerKind {
  LFK_Array,
  LFK_Iterator,
  LFK_ReverseIterator,
  LFK_PseudoArray
};

/// A map used to walk the AST in reverse: maps child Stmt to parent Stmt.
typedef llvm::DenseMap<const clang::Stmt *, const clang::Stmt *> StmtParentMap;

/// A map used to walk the AST in reverse:
///  maps VarDecl to the to parent DeclStmt.
typedef llvm::DenseMap<const clang::VarDecl *, const clang::DeclStmt *>
    DeclParentMap;

/// A map used to track which variables have been removed by a refactoring pass.
/// It maps the parent ForStmt to the removed index variable's VarDecl.
typedef llvm::DenseMap<const clang::ForStmt *, const clang::VarDecl *>
    ReplacedVarsMap;

/// A map used to remember the variable names generated in a Stmt
typedef llvm::DenseMap<const clang::Stmt *, std::string>
    StmtGeneratedVarNameMap;

/// A vector used to store the AST subtrees of an Expr.
typedef llvm::SmallVector<const clang::Expr *, 16> ComponentVector;

/// Class used build the reverse AST properties needed to detect
/// name conflicts and free variables.
class StmtAncestorASTVisitor
    : public clang::RecursiveASTVisitor<StmtAncestorASTVisitor> {
public:
  StmtAncestorASTVisitor() { StmtStack.push_back(nullptr); }

  /// Run the analysis on the AST.
  ///
  /// In case we're running this analysis multiple times, don't repeat the work.
  void gatherAncestors(ASTContext &Ctx) {
    if (StmtAncestors.empty())
      TraverseAST(Ctx);
  }

  /// Accessor for StmtAncestors.
  const StmtParentMap &getStmtToParentStmtMap() { return StmtAncestors; }

  /// Accessor for DeclParents.
  const DeclParentMap &getDeclToParentStmtMap() { return DeclParents; }

  friend class clang::RecursiveASTVisitor<StmtAncestorASTVisitor>;

private:
  StmtParentMap StmtAncestors;
  DeclParentMap DeclParents;
  llvm::SmallVector<const clang::Stmt *, 16> StmtStack;

  bool TraverseStmt(clang::Stmt *Statement);
  bool VisitDeclStmt(clang::DeclStmt *Statement);
};

/// Class used to find the variables and member expressions on which an
/// arbitrary expression depends.
class ComponentFinderASTVisitor
    : public clang::RecursiveASTVisitor<ComponentFinderASTVisitor> {
public:
  ComponentFinderASTVisitor() = default;

  /// Find the components of an expression and place them in a ComponentVector.
  void findExprComponents(const clang::Expr *SourceExpr) {
    TraverseStmt(const_cast<clang::Expr *>(SourceExpr));
  }

  /// Accessor for Components.
  const ComponentVector &getComponents() { return Components; }

  friend class clang::RecursiveASTVisitor<ComponentFinderASTVisitor>;

private:
  ComponentVector Components;

  bool VisitDeclRefExpr(clang::DeclRefExpr *E);
  bool VisitMemberExpr(clang::MemberExpr *Member);
};

/// Class used to determine if an expression is dependent on a variable declared
/// inside of the loop where it would be used.
class DependencyFinderASTVisitor
    : public clang::RecursiveASTVisitor<DependencyFinderASTVisitor> {
public:
  DependencyFinderASTVisitor(const StmtParentMap *StmtParents,
                             const DeclParentMap *DeclParents,
                             const ReplacedVarsMap *ReplacedVars,
                             const clang::Stmt *ContainingStmt)
      : StmtParents(StmtParents), DeclParents(DeclParents),
        ContainingStmt(ContainingStmt), ReplacedVars(ReplacedVars) {}

  /// Run the analysis on Body, and return true iff the expression
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
  /// within the loop body.
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
class DeclFinderASTVisitor
    : public clang::RecursiveASTVisitor<DeclFinderASTVisitor> {
public:
  DeclFinderASTVisitor(const StringRef &Name,
                       const StmtGeneratedVarNameMap *GeneratedDecls)
      : Name(Name), GeneratedDecls(GeneratedDecls), Found(false) {}

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
  /// GeneratedDecls keeps track of ForStmts which have been transformed,
  /// mapping each modified ForStmt to the variable generated in the loop.
  const StmtGeneratedVarNameMap *GeneratedDecls;
  bool Found;

  bool VisitForStmt(clang::ForStmt *F);
  bool VisitNamedDecl(clang::NamedDecl *D);
  bool VisitDeclRefExpr(clang::DeclRefExpr *D);
  bool VisitTypeLoc(clang::TypeLoc TL);
};

/// The information needed to describe a valid convertible usage
/// of an array index or iterator.
struct Usage {
  enum UsageKind {
    // Regular usages of the loop index (the ones not specified below). Some
    // examples:
    // \code
    //   int X = 8 * Arr[i];
    //               ^~~~~~
    //   f(param1, param2, *It);
    //                     ^~~
    //   if (Vec[i].SomeBool) {}
    //       ^~~~~~
    // \endcode
    UK_Default,
    // Indicates whether this is an access to a member through the arrow
    // operator on pointers or iterators.
    UK_MemberThroughArrow,
    // If the variable is being captured by a lambda, indicates whether the
    // capture was done by value or by reference.
    UK_CaptureByCopy,
    UK_CaptureByRef
  };
  // The expression that is going to be converted. Null in case of lambda
  // captures.
  const Expr *Expression;

  UsageKind Kind;

  // Range that covers this usage.
  SourceRange Range;

  explicit Usage(const Expr *E)
      : Expression(E), Kind(UK_Default), Range(Expression->getSourceRange()) {}
  Usage(const Expr *E, UsageKind Kind, SourceRange Range)
      : Expression(E), Kind(Kind), Range(std::move(Range)) {}
};

/// A class to encapsulate lowering of the tool's confidence level.
class Confidence {
public:
  enum Level {
    // Transformations that are likely to change semantics.
    CL_Risky,

    // Transformations that might change semantics.
    CL_Reasonable,

    // Transformations that will not change semantics.
    CL_Safe
  };
  /// Initialize confidence level.
  explicit Confidence(Confidence::Level Level) : CurrentLevel(Level) {}

  /// Lower the internal confidence level to Level, but do not raise it.
  void lowerTo(Confidence::Level Level) {
    CurrentLevel = std::min(Level, CurrentLevel);
  }

  /// Return the internal confidence level.
  Level getLevel() const { return CurrentLevel; }

private:
  Level CurrentLevel;
};

// The main computational result of ForLoopIndexVisitor.
typedef llvm::SmallVector<Usage, 8> UsageResult;

// General functions used by ForLoopIndexUseVisitor and LoopConvertCheck.
const Expr *digThroughConstructorsConversions(const Expr *E);
bool areSameExpr(ASTContext *Context, const Expr *First, const Expr *Second);
const DeclRefExpr *getDeclRef(const Expr *E);
bool areSameVariable(const ValueDecl *First, const ValueDecl *Second);

/// Discover usages of expressions consisting of index or iterator
/// access.
///
/// Given an index variable, recursively crawls a for loop to discover if the
/// index variable is used in a way consistent with range-based for loop access.
class ForLoopIndexUseVisitor
    : public RecursiveASTVisitor<ForLoopIndexUseVisitor> {
public:
  ForLoopIndexUseVisitor(ASTContext *Context, const VarDecl *IndexVar,
                         const VarDecl *EndVar, const Expr *ContainerExpr,
                         const Expr *ArrayBoundExpr,
                         bool ContainerNeedsDereference);

  /// Finds all uses of IndexVar in Body, placing all usages in Usages,
  /// and returns true if IndexVar was only used in a way consistent with a
  /// range-based for loop.
  ///
  /// The general strategy is to reject any DeclRefExprs referencing IndexVar,
  /// with the exception of certain acceptable patterns.
  /// For arrays, the DeclRefExpr for IndexVar must appear as the index of an
  /// ArraySubscriptExpression. Iterator-based loops may dereference
  /// IndexVar or call methods through operator-> (builtin or overloaded).
  /// Array-like containers may use IndexVar as a parameter to the at() member
  /// function and in overloaded operator[].
  bool findAndVerifyUsages(const Stmt *Body);

  /// Add a set of components that we should consider relevant to the
  /// container.
  void addComponents(const ComponentVector &Components);

  /// Accessor for Usages.
  const UsageResult &getUsages() const { return Usages; }

  /// Adds the Usage if it was not added before.
  void addUsage(const Usage &U);

  /// Get the container indexed by IndexVar, if any.
  const Expr *getContainerIndexed() const { return ContainerExpr; }

  /// Returns the statement declaring the variable created as an alias
  /// for the loop element, if any.
  const DeclStmt *getAliasDecl() const { return AliasDecl; }

  /// Accessor for ConfidenceLevel.
  Confidence::Level getConfidenceLevel() const {
    return ConfidenceLevel.getLevel();
  }

  /// Indicates if the alias declaration was in a place where it cannot
  /// simply be removed but rather replaced with a use of the alias variable.
  /// For example, variables declared in the condition of an if, switch, or for
  /// stmt.
  bool aliasUseRequired() const { return ReplaceWithAliasUse; }

  /// Indicates if the alias declaration came from the init clause of a
  /// nested for loop. SourceRanges provided by Clang for DeclStmts in this
  /// case need to be adjusted.
  bool aliasFromForInit() const { return AliasFromForInit; }

private:
  /// Typedef used in CRTP functions.
  typedef RecursiveASTVisitor<ForLoopIndexUseVisitor> VisitorBase;
  friend class RecursiveASTVisitor<ForLoopIndexUseVisitor>;

  /// Overriden methods for RecursiveASTVisitor's traversal.
  bool TraverseArraySubscriptExpr(ArraySubscriptExpr *E);
  bool TraverseCXXMemberCallExpr(CXXMemberCallExpr *MemberCall);
  bool TraverseCXXOperatorCallExpr(CXXOperatorCallExpr *OpCall);
  bool TraverseLambdaCapture(LambdaExpr *LE, const LambdaCapture *C,
                             Expr *Init);
  bool TraverseMemberExpr(MemberExpr *Member);
  bool TraverseUnaryOperator(UnaryOperator *Uop);
  bool VisitDeclRefExpr(DeclRefExpr *E);
  bool VisitDeclStmt(DeclStmt *S);
  bool TraverseStmt(Stmt *S);

  /// Add an expression to the list of expressions on which the container
  /// expression depends.
  void addComponent(const Expr *E);

  // Input member variables:
  ASTContext *Context;
  /// The index variable's VarDecl.
  const VarDecl *IndexVar;
  /// The loop's 'end' variable, which cannot be mentioned at all.
  const VarDecl *EndVar;
  /// The Expr which refers to the container.
  const Expr *ContainerExpr;
  /// The Expr which refers to the terminating condition for array-based loops.
  const Expr *ArrayBoundExpr;
  bool ContainerNeedsDereference;

  // Output member variables:
  /// A container which holds all usages of IndexVar as the index of
  /// ArraySubscriptExpressions.
  UsageResult Usages;
  llvm::SmallSet<SourceLocation, 8> UsageLocations;
  bool OnlyUsedAsIndex;
  /// The DeclStmt for an alias to the container element.
  const DeclStmt *AliasDecl;
  Confidence ConfidenceLevel;
  /// A list of expressions on which ContainerExpr depends.
  ///
  /// If any of these expressions are encountered outside of an acceptable usage
  /// of the loop element, lower our confidence level.
  llvm::SmallVector<std::pair<const Expr *, llvm::FoldingSetNodeID>, 16>
      DependentExprs;

  /// The parent-in-waiting. Will become the real parent once we traverse down
  /// one level in the AST.
  const Stmt *NextStmtParent;
  /// The actual parent of a node when Visit*() calls are made. Only the
  /// parentage of DeclStmt's to possible iteration/selection statements is of
  /// importance.
  const Stmt *CurrStmtParent;

  /// \see aliasUseRequired().
  bool ReplaceWithAliasUse;
  /// \see aliasFromForInit().
  bool AliasFromForInit;
};

struct TUTrackingInfo {
  /// Reset and initialize per-TU tracking information.
  ///
  /// Must be called before using container accessors.
  TUTrackingInfo() : ParentFinder(new StmtAncestorASTVisitor) {}

  StmtAncestorASTVisitor &getParentFinder() { return *ParentFinder; }
  StmtGeneratedVarNameMap &getGeneratedDecls() { return GeneratedDecls; }
  ReplacedVarsMap &getReplacedVars() { return ReplacedVars; }

private:
  std::unique_ptr<StmtAncestorASTVisitor> ParentFinder;
  StmtGeneratedVarNameMap GeneratedDecls;
  ReplacedVarsMap ReplacedVars;
};

/// Create names for generated variables within a particular statement.
///
/// VariableNamer uses a DeclContext as a reference point, checking for any
/// conflicting declarations higher up in the context or within SourceStmt.
/// It creates a variable name using hints from a source container and the old
/// index, if they exist.
class VariableNamer {
public:
  // Supported naming styles.
  enum NamingStyle {
    NS_CamelBack,
    NS_CamelCase,
    NS_LowerCase,
    NS_UpperCase,
  };

  VariableNamer(StmtGeneratedVarNameMap *GeneratedDecls,
                const StmtParentMap *ReverseAST, const clang::Stmt *SourceStmt,
                const clang::VarDecl *OldIndex,
                const clang::ValueDecl *TheContainer,
                const clang::ASTContext *Context, NamingStyle Style)
      : GeneratedDecls(GeneratedDecls), ReverseAST(ReverseAST),
        SourceStmt(SourceStmt), OldIndex(OldIndex), TheContainer(TheContainer),
        Context(Context), Style(Style) {}

  /// Generate a new index name.
  ///
  /// Generates the name to be used for an inserted iterator. It relies on
  /// declarationExists() to determine that there are no naming conflicts, and
  /// tries to use some hints from the container name and the old index name.
  std::string createIndexName();

private:
  StmtGeneratedVarNameMap *GeneratedDecls;
  const StmtParentMap *ReverseAST;
  const clang::Stmt *SourceStmt;
  const clang::VarDecl *OldIndex;
  const clang::ValueDecl *TheContainer;
  const clang::ASTContext *Context;
  const NamingStyle Style;

  // Determine whether or not a declaration that would conflict with Symbol
  // exists in an outer context or in any statement contained in SourceStmt.
  bool declarationExists(llvm::StringRef Symbol);
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_LOOP_CONVERT_UTILS_H
