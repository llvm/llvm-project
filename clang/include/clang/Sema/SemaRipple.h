//===----- SemaRipple.h - Semantic Analysis for Ripple constructs ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file declares semantic analysis for Ripple constructs and clauses.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMARIPPLE_H
#define LLVM_CLANG_SEMA_SEMARIPPLE_H

#include "clang/AST/StmtRipple.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/SemaBase.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace clang {

class CallExpr;
class Expr;
class FunctionDecl;
class IdentifierInfo;
class Sema;
class Stmt;
class ValueDecl;

class SemaRipple : public SemaBase {
public:
  SemaRipple(Sema &S) : SemaBase(S) {};

  /// Checks if @p Op can be converted to Integer type w/ error reporting
  ExprResult PerformRippleImplicitIntegerConversion(SourceLocation Loc,
                                                    Expr *Op);

  /// Constructor of a Ripple parallel compute construct statement
  StmtResult
  CreateRippleParallelComputeStmt(SourceRange PragmaLoc, SourceRange PELoc,
                                  SourceRange DimsLoc, ValueDecl *BlockShape,
                                  ArrayRef<uint64_t> Dims,
                                  Stmt *AssociatedStatement, bool NoRemainder);

  // Checks that dimension indices are uniq
  void ActOnDuplicateDimensionIndex(const RippleComputeConstruct &S);

  /// Checks Ripple builtin calls
  bool CheckBuiltinFunctionCall(const FunctionDecl *FDecl, unsigned BuiltinID,
                                const CallExpr *RippleBICall);

  bool CheckHasRippleBlockType(const Expr *E, unsigned BuiltinID);

  struct AnnotationData {
    SourceRange BlockShapeRange;
    SourceRange DimsRange;
    IdentifierInfo *BlockShape;
    SmallVector<uint64_t, 4> Dims;
    bool IgnoreNullStatements = false;
    bool NoRemainder = false;
  };
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMARIPPLE_H
