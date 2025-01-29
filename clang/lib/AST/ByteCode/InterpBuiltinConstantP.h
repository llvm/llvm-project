//===-------------------- InterpBuiltinConstantP.h --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This is an implementation of __builtin_constant_p.
//
// __builtin_constant_p is a GCC extension that is supposed to return 1 if the
// given expression can be evaluated at compile time. This is a problem for our
// byte code approach, however.
//
// 1) We cannot just keep the expression unevaluated and create a new
//    Compiler<EvalEmitter> when evaluating the bcp call. This doesn't work
//    because the expression can refer to variables from the current InterpFrame
//    or parameters from the function, etc.
//
// 2) We have no mechanism to suppress diagnostics and side-effects and to
//    eventually just record whether the evaluation of an expression was
//    successful or not, in byte code. If the evaluation fails, it
//    fails--and will never reach the end of the bcp call. This COULD be
//    changed, but that means changing how byte code is interpreted
//    everywhere, just because of one builtin.
//
// So, here we implement our own Visitor that basically implements a subset of
// working operations for the expression passed to __builtin_constant_p.
//
// While it is easy to come up with examples where we don't behave correctly,
// __builtin_constant_p is usually used to check whether a single parameter
// or variable is known at compile time, so the expressions used in reality
// are very simple.

#ifndef LLVM_CLANG_AST_INTERP_BUILTIN_CONSTANT_P_H
#define LLVM_CLANG_AST_INTERP_BUILTIN_CONSTANT_P_H

#include "InterpState.h"
#include "Pointer.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"

namespace clang {
namespace interp {

class BCPVisitor final : public DynamicRecursiveASTVisitor {
public:
  BCPVisitor(InterpState &S) : Frame(S.Current), S(S) {}

  bool VisitStmt(Stmt *S) override;
  bool VisitDeclRefExpr(DeclRefExpr *E) override;
  bool VisitUnaryOperator(UnaryOperator *E) override;
  bool VisitBinaryOperator(BinaryOperator *E) override;
  bool VisitCastExpr(CastExpr *E) override;
  bool VisitCallExpr(CallExpr *E) override;
  bool VisitConditionalOperator(ConditionalOperator *E) override;
  bool VisitMemberExpr(MemberExpr *E) override;

  bool Result = true;

private:
  InterpFrame *Frame;
  InterpState &S;

  std::optional<unsigned> findLocal(const ValueDecl *D) const;
  std::optional<unsigned> findParam(const ValueDecl *D) const;
  bool pointerChainIsLive(const Pointer &P) const;
  Pointer getPointer(const ValueDecl *D) const;

  bool Fail() {
    Result = false;
    return false;
  }
};

} // namespace interp
} // namespace clang
#endif
