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
#include "clang/AST/Attr.h"
#include "clang/AST/OpenACCClause.h"

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

OpenACCRoutineDecl *
OpenACCRoutineDecl::Create(ASTContext &Ctx, DeclContext *DC,
                           SourceLocation StartLoc, SourceLocation DirLoc,
                           SourceLocation LParenLoc, Expr *FuncRef,
                           SourceLocation RParenLoc, SourceLocation EndLoc,
                           ArrayRef<const OpenACCClause *> Clauses) {
  return new (Ctx, DC,
              additionalSizeToAlloc<const OpenACCClause *>(Clauses.size()))
      OpenACCRoutineDecl(DC, StartLoc, DirLoc, LParenLoc, FuncRef, RParenLoc,
                         EndLoc, Clauses);
}

OpenACCRoutineDecl *
OpenACCRoutineDecl::CreateDeserialized(ASTContext &Ctx, GlobalDeclID ID,
                                       unsigned NumClauses) {
  return new (Ctx, ID, additionalSizeToAlloc<const OpenACCClause *>(NumClauses))
      OpenACCRoutineDecl(NumClauses);
}

void OpenACCRoutineDeclAttr::printPrettyPragma(
    llvm::raw_ostream &OS, const clang::PrintingPolicy &P) const {
  if (Clauses.size() > 0) {
    OS << ' ';
    OpenACCClausePrinter Printer{OS, P};
    Printer.VisitClauseList(Clauses);
  }
}
