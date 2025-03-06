//===--- DeclOpenACC.cpp - Classes for OpenACC Constructs -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the subclasses of Decl class declared in Decl.h
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclOpenACC.h"
#include "clang/AST/ASTContext.h"

using namespace clang;

OpenACCDeclareDecl *
OpenACCDeclareDecl::Create(ASTContext &Ctx, DeclContext *DC,
                           SourceLocation StartLoc, SourceLocation DirLoc,
                           SourceLocation EndLoc,
                           ArrayRef<const OpenACCClause *> Clauses) {
  return new (Ctx, DC,
              additionalSizeToAlloc<const OpenACCClause *>(Clauses.size()))
      OpenACCDeclareDecl(DC, StartLoc, DirLoc, EndLoc, Clauses);
}

OpenACCDeclareDecl *
OpenACCDeclareDecl::CreateDeserialized(ASTContext &Ctx, GlobalDeclID ID,
                                       unsigned NumClauses) {
  return new (Ctx, ID, additionalSizeToAlloc<const OpenACCClause *>(NumClauses))
      OpenACCDeclareDecl(NumClauses);
}
