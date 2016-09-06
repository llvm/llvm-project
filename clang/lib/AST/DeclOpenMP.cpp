//===--- DeclOpenMP.cpp - Declaration OpenMP AST Node Implementation ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief This file implements OMPThreadPrivateDecl, OMPCapturedExprDecl
/// classes.
///
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/Expr.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// OMPThreadPrivateDecl Implementation.
//===----------------------------------------------------------------------===//

void OMPThreadPrivateDecl::anchor() { }

OMPThreadPrivateDecl *OMPThreadPrivateDecl::Create(ASTContext &C,
                                                   DeclContext *DC,
                                                   SourceLocation L,
                                                   ArrayRef<Expr *> VL) {
  OMPThreadPrivateDecl *D =
      new (C, DC, additionalSizeToAlloc<Expr *>(VL.size()))
          OMPThreadPrivateDecl(OMPThreadPrivate, DC, L);
  D->NumVars = VL.size();
  D->setVars(VL);
  return D;
}

OMPThreadPrivateDecl *OMPThreadPrivateDecl::CreateDeserialized(ASTContext &C,
                                                               unsigned ID,
                                                               unsigned N) {
  OMPThreadPrivateDecl *D = new (C, ID, additionalSizeToAlloc<Expr *>(N))
      OMPThreadPrivateDecl(OMPThreadPrivate, nullptr, SourceLocation());
  D->NumVars = N;
  return D;
}

void OMPThreadPrivateDecl::setVars(ArrayRef<Expr *> VL) {
  assert(VL.size() == NumVars &&
         "Number of variables is not the same as the preallocated buffer");
  std::uninitialized_copy(VL.begin(), VL.end(), getTrailingObjects<Expr *>());
}

//===----------------------------------------------------------------------===//
// OMPDeclareReductionDecl Implementation.
//===----------------------------------------------------------------------===//

void OMPDeclareReductionDecl::anchor() {}

OMPDeclareReductionDecl *OMPDeclareReductionDecl::Create(
    ASTContext &C, DeclContext *DC, SourceLocation L, DeclarationName Name,
    QualType T, OMPDeclareReductionDecl *PrevDeclInScope) {
  return new (C, DC) OMPDeclareReductionDecl(OMPDeclareReduction, DC, L, Name,
                                             T, PrevDeclInScope);
}

OMPDeclareReductionDecl *
OMPDeclareReductionDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  return new (C, ID) OMPDeclareReductionDecl(
      OMPDeclareReduction, /*DC=*/nullptr, SourceLocation(), DeclarationName(),
      QualType(), /*PrevDeclInScope=*/nullptr);
}

OMPDeclareReductionDecl *OMPDeclareReductionDecl::getPrevDeclInScope() {
  return cast_or_null<OMPDeclareReductionDecl>(
      PrevDeclInScope.get(getASTContext().getExternalSource()));
}
const OMPDeclareReductionDecl *
OMPDeclareReductionDecl::getPrevDeclInScope() const {
  return cast_or_null<OMPDeclareReductionDecl>(
      PrevDeclInScope.get(getASTContext().getExternalSource()));
}

//===----------------------------------------------------------------------===//
// OMPCapturedExprDecl Implementation.
//===----------------------------------------------------------------------===//

void OMPCapturedExprDecl::anchor() {}

OMPCapturedExprDecl *OMPCapturedExprDecl::Create(ASTContext &C, DeclContext *DC,
                                                 IdentifierInfo *Id,
                                                 QualType T) {
  return new (C, DC) OMPCapturedExprDecl(C, DC, Id, T);
}

OMPCapturedExprDecl *OMPCapturedExprDecl::CreateDeserialized(ASTContext &C,
                                                             unsigned ID) {
  return new (C, ID) OMPCapturedExprDecl(C, nullptr, nullptr, QualType());
}

