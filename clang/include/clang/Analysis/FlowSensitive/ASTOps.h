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
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
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

/// Returns the fields of a `RecordDecl` that are initialized by an
/// `InitListExpr`, in the order in which they appear in
/// `InitListExpr::inits()`.
/// `Init->getType()` must be a record type.
std::vector<const FieldDecl *>
getFieldsForInitListExpr(const InitListExpr *InitList);

// A collection of several types of declarations, all referenced from the same
// function.
struct ReferencedDecls {
  // Fields includes non-static data members.
  FieldSet Fields;
  // Globals includes all variables with global storage, notably including
  // static data members and static variables declared within a function.
  llvm::DenseSet<const VarDecl *> Globals;
  // Functions includes free functions and member functions which are
  // referenced, but not necessarily called.
  llvm::DenseSet<const FunctionDecl *> Functions;
};

/// Collects and returns fields, global variables and functions that are
/// declared in or referenced from `FD`.
ReferencedDecls getReferencedDecls(const FunctionDecl &FD);
} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_ASTOPS_H
