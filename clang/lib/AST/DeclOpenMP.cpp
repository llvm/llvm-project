//===--- DeclOpenMP.cpp - Declaration OpenMP AST Node Implementation ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements OMPThreadPrivateDecl, OMPCapturedExprDecl
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
// OMPAllocateDecl Implementation.
//===----------------------------------------------------------------------===//

void OMPAllocateDecl::anchor() { }

OMPAllocateDecl *OMPAllocateDecl::Create(ASTContext &C, DeclContext *DC,
                                         SourceLocation L,
                                         ArrayRef<Expr *> VL) {
  OMPAllocateDecl *D = new (C, DC, additionalSizeToAlloc<Expr *>(VL.size()))
      OMPAllocateDecl(OMPAllocate, DC, L);
  D->NumVars = VL.size();
  D->setVars(VL);
  return D;
}

OMPAllocateDecl *OMPAllocateDecl::CreateDeserialized(ASTContext &C, unsigned ID,
                                                     unsigned N) {
  OMPAllocateDecl *D = new (C, ID, additionalSizeToAlloc<Expr *>(N))
      OMPAllocateDecl(OMPAllocate, nullptr, SourceLocation());
  D->NumVars = N;
  return D;
}

void OMPAllocateDecl::setVars(ArrayRef<Expr *> VL) {
  assert(VL.size() == NumVars &&
         "Number of variables is not the same as the preallocated buffer");
  std::uninitialized_copy(VL.begin(), VL.end(), getTrailingObjects<Expr *>());
}

//===----------------------------------------------------------------------===//
// OMPRequiresDecl Implementation.
//===----------------------------------------------------------------------===//

void OMPRequiresDecl::anchor() {}

OMPRequiresDecl *OMPRequiresDecl::Create(ASTContext &C, DeclContext *DC,
                                         SourceLocation L,
                                         ArrayRef<OMPClause *> CL) {
  OMPRequiresDecl *D =
      new (C, DC, additionalSizeToAlloc<OMPClause *>(CL.size()))
      OMPRequiresDecl(OMPRequires, DC, L);
  D->NumClauses = CL.size();
  D->setClauses(CL);
  return D;
}

OMPRequiresDecl *OMPRequiresDecl::CreateDeserialized(ASTContext &C, unsigned ID,
                                                     unsigned N) {
  OMPRequiresDecl *D = new (C, ID, additionalSizeToAlloc<OMPClause *>(N))
      OMPRequiresDecl(OMPRequires, nullptr, SourceLocation());
  D->NumClauses = N;
  return D;
}

void OMPRequiresDecl::setClauses(ArrayRef<OMPClause *> CL) {
  assert(CL.size() == NumClauses &&
         "Number of clauses is not the same as the preallocated buffer");
  std::uninitialized_copy(CL.begin(), CL.end(),
                          getTrailingObjects<OMPClause *>());
}

//===----------------------------------------------------------------------===//
// OMPDeclareReductionDecl Implementation.
//===----------------------------------------------------------------------===//

OMPDeclareReductionDecl::OMPDeclareReductionDecl(
    Kind DK, DeclContext *DC, SourceLocation L, DeclarationName Name,
    QualType Ty, OMPDeclareReductionDecl *PrevDeclInScope)
    : ValueDecl(DK, DC, L, Name, Ty), DeclContext(DK), Combiner(nullptr),
      PrevDeclInScope(PrevDeclInScope) {
  setInitializer(nullptr, CallInit);
}

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
// OMPDeclareMapperDecl Implementation.
//===----------------------------------------------------------------------===//

void OMPDeclareMapperDecl::anchor() {}

OMPDeclareMapperDecl *
OMPDeclareMapperDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                             DeclarationName Name, QualType T,
                             DeclarationName VarName,
                             OMPDeclareMapperDecl *PrevDeclInScope) {
  return new (C, DC) OMPDeclareMapperDecl(OMPDeclareMapper, DC, L, Name, T,
                                          VarName, PrevDeclInScope);
}

OMPDeclareMapperDecl *OMPDeclareMapperDecl::CreateDeserialized(ASTContext &C,
                                                               unsigned ID,
                                                               unsigned N) {
  auto *D = new (C, ID)
      OMPDeclareMapperDecl(OMPDeclareMapper, /*DC=*/nullptr, SourceLocation(),
                           DeclarationName(), QualType(), DeclarationName(),
                           /*PrevDeclInScope=*/nullptr);
  if (N) {
    auto **ClauseStorage = C.Allocate<OMPClause *>(N);
    D->Clauses = llvm::makeMutableArrayRef<OMPClause *>(ClauseStorage, N);
  }
  return D;
}

/// Creates an array of clauses to this mapper declaration and intializes
/// them. The space used to store clause pointers is dynamically allocated,
/// because we do not know the number of clauses when creating
/// OMPDeclareMapperDecl
void OMPDeclareMapperDecl::CreateClauses(ASTContext &C,
                                         ArrayRef<OMPClause *> CL) {
  assert(Clauses.empty() && "Number of clauses should be 0 on initialization");
  size_t NumClauses = CL.size();
  if (NumClauses) {
    auto **ClauseStorage = C.Allocate<OMPClause *>(NumClauses);
    Clauses = llvm::makeMutableArrayRef<OMPClause *>(ClauseStorage, NumClauses);
    setClauses(CL);
  }
}

void OMPDeclareMapperDecl::setClauses(ArrayRef<OMPClause *> CL) {
  assert(CL.size() == Clauses.size() &&
         "Number of clauses is not the same as the preallocated buffer");
  std::uninitialized_copy(CL.begin(), CL.end(), Clauses.data());
}

OMPDeclareMapperDecl *OMPDeclareMapperDecl::getPrevDeclInScope() {
  return cast_or_null<OMPDeclareMapperDecl>(
      PrevDeclInScope.get(getASTContext().getExternalSource()));
}

const OMPDeclareMapperDecl *OMPDeclareMapperDecl::getPrevDeclInScope() const {
  return cast_or_null<OMPDeclareMapperDecl>(
      PrevDeclInScope.get(getASTContext().getExternalSource()));
}

//===----------------------------------------------------------------------===//
// OMPCapturedExprDecl Implementation.
//===----------------------------------------------------------------------===//

void OMPCapturedExprDecl::anchor() {}

OMPCapturedExprDecl *OMPCapturedExprDecl::Create(ASTContext &C, DeclContext *DC,
                                                 IdentifierInfo *Id, QualType T,
                                                 SourceLocation StartLoc) {
  return new (C, DC) OMPCapturedExprDecl(
      C, DC, Id, T, C.getTrivialTypeSourceInfo(T), StartLoc);
}

OMPCapturedExprDecl *OMPCapturedExprDecl::CreateDeserialized(ASTContext &C,
                                                             unsigned ID) {
  return new (C, ID) OMPCapturedExprDecl(C, nullptr, nullptr, QualType(),
                                         /*TInfo=*/nullptr, SourceLocation());
}

SourceRange OMPCapturedExprDecl::getSourceRange() const {
  assert(hasInit());
  return SourceRange(getInit()->getBeginLoc(), getInit()->getEndLoc());
}
