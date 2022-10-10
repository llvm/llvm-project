//===- SemaHLSL.cpp - Semantic Analysis for HLSL constructs ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This implements Semantic Analysis for HLSL constructs.
//===----------------------------------------------------------------------===//

#include "clang/Sema/Sema.h"

using namespace clang;

Decl *Sema::ActOnStartHLSLBuffer(Scope *BufferScope, bool CBuffer,
                                 SourceLocation KwLoc, IdentifierInfo *Ident,
                                 SourceLocation IdentLoc,
                                 SourceLocation LBrace) {
  // For anonymous namespace, take the location of the left brace.
  DeclContext *LexicalParent = getCurLexicalContext();
  HLSLBufferDecl *Result = HLSLBufferDecl::Create(
      Context, LexicalParent, CBuffer, KwLoc, Ident, IdentLoc, LBrace);

  PushOnScopeChains(Result, BufferScope);
  PushDeclContext(BufferScope, Result);

  return Result;
}

void Sema::ActOnFinishHLSLBuffer(Decl *Dcl, SourceLocation RBrace) {
  auto *BufDecl = cast<HLSLBufferDecl>(Dcl);
  BufDecl->setRBraceLoc(RBrace);
  PopDeclContext();
}
