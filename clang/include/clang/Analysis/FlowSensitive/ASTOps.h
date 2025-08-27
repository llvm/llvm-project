//===-- ASTOps.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Operations on AST nodes that are used in flow-sensitive analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_ASTOPS_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_ASTOPS_H

#include "clang/AST/Decl.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/TypeBase.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

namespace clang {
namespace dataflow {

/// Skip past nodes that the CFG does not emit. These nodes are invisible to
/// flow-sensitive analysis, and should be ignored as they will effectively not
/// exist.
///
///   * `ParenExpr` - The CFG takes the operator precedence into account, but
///   otherwise omits the node afterwards.
///
///   * `ExprWithCleanups` - The CFG will generate the appropriate calls to
///   destructors and then omit the node.
///
const Expr &ignoreCFGOmittedNodes(const Expr &E);
const Stmt &ignoreCFGOmittedNodes(const Stmt &S);

/// A set of `FieldDecl *`. Use `SmallSetVector` to guarantee deterministic
/// iteration order.
using FieldSet = llvm::SmallSetVector<const FieldDecl *, 4>;

/// Returns the set of all fields in the type.
FieldSet getObjectFields(QualType Type);

/// Returns whether `Fields` and `FieldLocs` contain the same fields.
bool containsSameFields(const FieldSet &Fields,
                        const RecordStorageLocation::FieldToLoc &FieldLocs);

/// Helper class for initialization of a record with an `InitListExpr`.
/// `InitListExpr::inits()` contains the initializers for both the base classes
/// and the fields of the record; this helper class separates these out into two
/// different lists. In addition, it deals with special cases associated with
/// unions.
class RecordInitListHelper {
public:
  // `InitList` must have record type.
  RecordInitListHelper(const InitListExpr *InitList);
  RecordInitListHelper(const CXXParenListInitExpr *ParenInitList);

  // Base classes with their associated initializer expressions.
  ArrayRef<std::pair<const CXXBaseSpecifier *, Expr *>> base_inits() const {
    return BaseInits;
  }

  // Fields with their associated initializer expressions.
  ArrayRef<std::pair<const FieldDecl *, Expr *>> field_inits() const {
    return FieldInits;
  }

private:
  RecordInitListHelper(QualType Ty, std::vector<const FieldDecl *> Fields,
                       ArrayRef<Expr *> Inits);

  SmallVector<std::pair<const CXXBaseSpecifier *, Expr *>> BaseInits;
  SmallVector<std::pair<const FieldDecl *, Expr *>> FieldInits;

  // We potentially synthesize an `ImplicitValueInitExpr` for unions. It's a
  // member variable because we store a pointer to it in `FieldInits`.
  std::optional<ImplicitValueInitExpr> ImplicitValueInitForUnion;
};

/// Specialization of `RecursiveASTVisitor` that visits those nodes that are
/// relevant to the dataflow analysis; generally, these are the ones that also
/// appear in the CFG.
/// To start the traversal, call `TraverseStmt()` on the statement or body of
/// the function to analyze. Don't call `TraverseDecl()` on the function itself;
/// this won't work as `TraverseDecl()` contains code to avoid traversing nested
/// functions.
class AnalysisASTVisitor : public DynamicRecursiveASTVisitor {
public:
  AnalysisASTVisitor() {
    ShouldVisitImplicitCode = true;
    ShouldVisitLambdaBody = false;
  }

  bool TraverseDecl(Decl *D) override {
    // Don't traverse nested record or function declarations.
    // - We won't be analyzing code contained in these anyway
    // - We don't model fields that are used only in these nested declaration,
    //   so trying to propagate a result object to initializers of such fields
    //   would cause an error.
    if (isa_and_nonnull<RecordDecl>(D) || isa_and_nonnull<FunctionDecl>(D))
      return true;

    return DynamicRecursiveASTVisitor::TraverseDecl(D);
  }

  // Don't traverse expressions in unevaluated contexts, as we don't model
  // fields that are only used in these.
  // Note: The operand of the `noexcept` operator is an unevaluated operand, but
  // nevertheless it appears in the Clang CFG, so we don't exclude it here.
  bool TraverseDecltypeTypeLoc(DecltypeTypeLoc,
                               bool TraverseQualifier) override {
    return true;
  }
  bool TraverseTypeOfExprTypeLoc(TypeOfExprTypeLoc,
                                 bool TraverseQualifier) override {
    return true;
  }
  bool TraverseCXXTypeidExpr(CXXTypeidExpr *TIE) override {
    if (TIE->isPotentiallyEvaluated())
      return DynamicRecursiveASTVisitor::TraverseCXXTypeidExpr(TIE);
    return true;
  }
  bool TraverseUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *) override {
    return true;
  }

  bool TraverseBindingDecl(BindingDecl *BD) override {
    // `RecursiveASTVisitor` doesn't traverse holding variables for
    // `BindingDecl`s by itself, so we need to tell it to.
    if (VarDecl *HoldingVar = BD->getHoldingVar())
      TraverseDecl(HoldingVar);
    return DynamicRecursiveASTVisitor::TraverseBindingDecl(BD);
  }
};

/// A collection of several types of declarations, all referenced from the same
/// function.
struct ReferencedDecls {
  /// Non-static member variables.
  FieldSet Fields;
  /// All variables with static storage duration, notably including static
  /// member variables and static variables declared within a function.
  llvm::DenseSet<const VarDecl *> Globals;
  /// Local variables, not including parameters or static variables declared
  /// within a function.
  llvm::DenseSet<const VarDecl *> Locals;
  /// Free functions and member functions which are referenced (but not
  /// necessarily called).
  llvm::DenseSet<const FunctionDecl *> Functions;
  /// When analyzing a lambda's call operator, the set of all parameters (from
  /// the surrounding function) that the lambda captures. Captured local
  /// variables are already included in `Locals` above.
  llvm::DenseSet<const ParmVarDecl *> LambdaCapturedParams;
};

/// Returns declarations that are declared in or referenced from `FD`.
ReferencedDecls getReferencedDecls(const FunctionDecl &FD);

/// Returns declarations that are declared in or referenced from `S`.
ReferencedDecls getReferencedDecls(const Stmt &S);

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_ASTOPS_H
