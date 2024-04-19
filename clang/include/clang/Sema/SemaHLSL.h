//===----- SemaHLSL.h ----- Semantic Analysis for HLSL constructs ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis for HLSL constructs.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMAHLSL_H
#define LLVM_CLANG_SEMA_SEMAHLSL_H

#include "clang/AST/DeclBase.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/SemaBase.h"

namespace clang {

class SemaHLSL : public SemaBase {
public:
  SemaHLSL(Sema &S);

  Decl *ActOnStartHLSLBuffer(Scope *BufferScope, bool CBuffer,
                             SourceLocation KwLoc, IdentifierInfo *Ident,
                             SourceLocation IdentLoc, SourceLocation LBrace);
  void ActOnFinishHLSLBuffer(Decl *Dcl, SourceLocation RBrace);
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMAHLSL_H
