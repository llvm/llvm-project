//===--- OverflowHandler.h - Overflow behavior handler labels -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers shared by Sema, CodeGen, and the CFG to decide which operations
// perform a trap overflow behavior type check and which label such a check
// branches to.
//
// A checked operation branches to the innermost visible handler label: a GNU
// local label (__label__) with the configured name declared in an enclosing
// compound statement, else a function-level label with that name. Lambdas,
// blocks, and other nested function bodies have their own handlers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_OVERFLOWHANDLER_H
#define LLVM_CLANG_AST_OVERFLOWHANDLER_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace clang {

class ASTContext;
class Decl;
class Expr;
class IdentifierInfo;
class LabelDecl;
class Stmt;
class VarDecl;

/// Returns the identifier of the overflow handler label, as configured by
/// -foverflow-behavior-label=.
IdentifierInfo *getOverflowHandlerName(ASTContext &Ctx);

/// Returns true if \p E is an operation on which a trap overflow behavior
/// type check is emitted, so that it may branch to an overflow handler.
bool isOverflowHandlerCheckedOp(ASTContext &Ctx, const Expr *E);

/// Returns true if \p S contains an operation for which
/// isOverflowHandlerCheckedOp is true and that branches to the handler in
/// effect at \p S. Nested lambda bodies and nested statements that introduce
/// their own handler are not searched, nor statement expressions unless
/// \p SearchStmtExprs.
bool containsOverflowHandlerCheckedOp(ASTContext &Ctx, const Stmt *S,
                                      bool SearchStmtExprs = true);

/// Returns the function-level (non-local) overflow handler label defined in
/// the body of \p CodeDecl (a function, block, or captured declaration), or
/// nullptr.
LabelDecl *findFunctionOverflowHandler(ASTContext &Ctx, const Decl *CodeDecl);

/// Returns the overflow handler that \p S introduces for the operations within
/// it: a GNU local label with the handler name declared in a compound
/// statement. Returns nullptr if checks
/// within \p S trap regardless of the enclosing handler: \p S is a captured
/// statement such as an OpenMP region, a __finally block, the body of a defer
/// statement, or an OpenACC compute construct. Returns std::nullopt if \p S
/// introduces no handler.
std::optional<LabelDecl *> getOverflowHandlerIntroducedBy(ASTContext &Ctx,
                                                          const Stmt *S);

/// Returns true if failed checks in the initializer of \p VD trap regardless
/// of the enclosing handler: the dynamic initializer of a static or
/// thread-local local variable runs under a guard that a branch would leave
/// held.
bool initializerIgnoresOverflowHandler(const VarDecl *VD);

/// Returns true if failed checks in \p Child, a child of \p Parent, trap
/// regardless of the enclosing handler: the initializer of a new-expression and
/// the operand of a throw-expression run after their storage is allocated,
/// which a branch would leak, and an __except filter is emitted apart from the
/// function body.
bool childIgnoresOverflowHandler(const Stmt *Parent, const Stmt *Child);

/// Returns true if \p Child, a child of \p Parent, is not evaluated: the
/// operand of sizeof or _Countof (unless it has variable length array type),
/// alignof, noexcept, or a non-polymorphic typeid, and the operands of
/// _Generic and __builtin_choose_expr that are not selected. No check in it
/// is emitted.
bool isOverflowHandlerUnevaluatedChild(ASTContext &Ctx, const Stmt *Parent,
                                       const Stmt *Child);

/// While alive, pushes onto \p Handlers the overflow handler that a statement
/// introduces, if any, or an explicit handler. A null entry makes checks trap.
/// Every walker that tracks the handler in scope uses this, so that they agree
/// on which label each operation branches to.
class OverflowHandlerScope {
  SmallVectorImpl<LabelDecl *> *Handlers = nullptr;

public:
  OverflowHandlerScope(SmallVectorImpl<LabelDecl *> &Handlers, ASTContext &Ctx,
                       const Stmt *S);
  OverflowHandlerScope(SmallVectorImpl<LabelDecl *> &Handlers,
                       LabelDecl *Handler)
      : Handlers(&Handlers) {
    Handlers.push_back(Handler);
  }
  ~OverflowHandlerScope() {
    if (Handlers)
      Handlers->pop_back();
  }
  /// Returns true if an entry was pushed.
  bool pushed() const { return Handlers; }
  OverflowHandlerScope(const OverflowHandlerScope &) = delete;
  OverflowHandlerScope &operator=(const OverflowHandlerScope &) = delete;
};

} // namespace clang

#endif // LLVM_CLANG_AST_OVERFLOWHANDLER_H
