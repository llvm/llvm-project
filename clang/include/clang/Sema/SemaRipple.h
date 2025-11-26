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

#include "clang/Sema/SemaBase.h"

namespace clang {

class CallExpr;
class Expr;
class FunctionDecl;
class Sema;

class SemaRipple : public SemaBase {
public:
  SemaRipple(Sema &S) : SemaBase(S) {};

  /// Checks Ripple builtin calls
  bool CheckBuiltinFunctionCall(const FunctionDecl *FDecl, unsigned BuiltinID,
                                const CallExpr *RippleBICall);

  bool CheckHasRippleBlockType(const Expr *E, unsigned BuiltinID);
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMARIPPLE_H
