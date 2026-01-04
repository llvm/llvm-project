//===----- SemaDirectX.h ----- Semantic Analysis for DirectX constructs----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis for DirectX constructs.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMADIRECTX_H
#define LLVM_CLANG_SEMA_SEMADIRECTX_H

#include "clang/AST/ASTFwd.h"
#include "clang/Sema/SemaBase.h"

namespace clang {
class SemaDirectX : public SemaBase {
public:
  SemaDirectX(Sema &S);

  bool CheckDirectXBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
};
} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMADIRECTX_H
