//===----- SemaCUDA.h ----- Semantic Analysis for C++20 coroutines --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis for C++20 coroutines.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMACOROUTINE_H
#define LLVM_CLANG_SEMA_SEMACOROUTINE_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/SemaBase.h"

namespace clang {
class SemaCoroutine : public SemaBase {
public:
  SemaCoroutine(Sema &S);

  /// The C++ "std::coroutine_traits" template, which is defined in
  /// \<coroutine_traits>
  ClassTemplateDecl *StdCoroutineTraitsCache;

  bool ActOnCoroutineBodyStart(Scope *S, SourceLocation KwLoc,
                               StringRef Keyword);
  ExprResult ActOnCoawaitExpr(Scope *S, SourceLocation KwLoc, Expr *E);
  ExprResult ActOnCoyieldExpr(Scope *S, SourceLocation KwLoc, Expr *E);
  StmtResult ActOnCoreturnStmt(Scope *S, SourceLocation KwLoc, Expr *E);

  ExprResult BuildOperatorCoawaitLookupExpr(Scope *S, SourceLocation Loc);
  ExprResult BuildOperatorCoawaitCall(SourceLocation Loc, Expr *E,
                                      UnresolvedLookupExpr *Lookup);
  ExprResult BuildResolvedCoawaitExpr(SourceLocation KwLoc, Expr *Operand,
                                      Expr *Awaiter, bool IsImplicit = false);
  ExprResult BuildUnresolvedCoawaitExpr(SourceLocation KwLoc, Expr *Operand,
                                        UnresolvedLookupExpr *Lookup);
  ExprResult BuildCoyieldExpr(SourceLocation KwLoc, Expr *E);
  StmtResult BuildCoreturnStmt(SourceLocation KwLoc, Expr *E,
                               bool IsImplicit = false);
  StmtResult BuildCoroutineBodyStmt(CoroutineBodyStmt::CtorArgs);
  bool buildCoroutineParameterMoves(SourceLocation Loc);
  VarDecl *buildCoroutinePromise(SourceLocation Loc);
  void CheckCompletedCoroutineBody(FunctionDecl *FD, Stmt *&Body);

  // As a clang extension, enforces that a non-coroutine function must be marked
  // with [[clang::coro_wrapper]] if it returns a type marked with
  // [[clang::coro_return_type]].
  // Expects that FD is not a coroutine.
  void CheckCoroutineWrapper(FunctionDecl *FD);
  /// Lookup 'coroutine_traits' in std namespace and std::experimental
  /// namespace. The namespace found is recorded in Namespace.
  ClassTemplateDecl *lookupCoroutineTraits(SourceLocation KwLoc,
                                           SourceLocation FuncLoc);
  /// Check that the expression co_await promise.final_suspend() shall not be
  /// potentially-throwing.
  bool checkFinalSuspendNoThrow(const Stmt *FinalSuspend);
};
} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMACOROUTINE_H
