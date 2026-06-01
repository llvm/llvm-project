//=- FlowNullability.h - Flow-sensitive null dereference checking -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines APIs for invoking flow-sensitive nullability analysis
// that detects dereferences of nullable pointers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_FLOWNULLABILITY_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_FLOWNULLABILITY_H

#include "clang/AST/Type.h"
#include "clang/Basic/Specifiers.h"

namespace clang {

class AnalysisDeclContext;
class CXXRecordDecl;
class Expr;
class FieldDecl;
class FunctionDecl;
class ParmVarDecl;
class VarDecl;

class FlowNullabilityHandler {
public:
  virtual ~FlowNullabilityHandler();
  virtual void handleNullableDereference(const Expr *DerefExpr,
                                         QualType PtrType) = 0;
  virtual void handleNullableArithmetic(const Expr *ArithExpr,
                                        QualType PtrType) {}
  virtual void handleNullableReturn(const Expr *ReturnExpr, QualType ExprType,
                                    QualType ReturnType) {}
  virtual void handleNullableAssignment(const Expr *AssignExpr,
                                        const VarDecl *LHSVar) {}
  virtual void handleNullableMemberAssignment(const Expr *AssignExpr,
                                              const FieldDecl *Member) {}
  virtual void handleNullableArgument(const Expr *ArgExpr,
                                      const ParmVarDecl *Param) {}

  /// Evidence collection: called when a pointer member is assigned.
  /// \p IsNonnull is true if the RHS is provably non-null.
  virtual void handleMemberAssignEvidence(const Expr *AssignExpr,
                                          const FieldDecl *Member,
                                          bool IsNonnull) {}

  /// Evidence collection: called when a function returns a pointer.
  /// \p IsNonnull is true if the returned expression is provably non-null.
  virtual void handleReturnEvidence(const Expr *RetExpr,
                                    const FunctionDecl *Func, bool IsNonnull) {}

  /// Evidence collection: called when a pointer argument is passed to a
  /// function parameter. \p IsNonnull is true if the argument is provably
  /// non-null at the call site.
  virtual void handleParameterEvidence(const Expr *ArgExpr,
                                       const ParmVarDecl *Param,
                                       const FunctionDecl *Func,
                                       bool IsNonnull) {}

  /// Summary evidence: called after the dataflow fixpoint when every
  /// return path in the function returns a provably non-null expression
  /// (address-of, this, new, narrowed var, etc.). Enables callers to
  /// treat the function's return as implicitly _Nonnull.
  virtual void handleAllReturnsNonnull(const FunctionDecl *Func) {}

  /// Query: has this function been previously analyzed and found to have
  /// all-returns-nonnull? Used by callers within the same TU to narrow
  /// returned pointers. Returns false by default (conservative).
  virtual bool isKnownAllReturnsNonnull(const FunctionDecl *Func) const {
    return false;
  }
};

void runFlowNullabilityAnalysis(AnalysisDeclContext &AC,
                                FlowNullabilityHandler &Handler,
                                bool StrictMode,
                                NullabilityKind DefaultNullability);

} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_FLOWNULLABILITY_H
