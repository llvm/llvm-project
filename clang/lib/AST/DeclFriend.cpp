//===- DeclFriend.cpp - C++ Friend Declaration AST Node Implementation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AST classes related to C++ friend
// declarations.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclFriend.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExternalASTSource.h"
#include <cassert>
#include <cstddef>

using namespace clang;

void FriendDecl::anchor() {}

FriendDecl *FriendDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                               FriendUnion Friend, SourceLocation FriendL,
                               SourceLocation EllipsisLoc) {
#ifndef NDEBUG
  if (const auto *D = dyn_cast<NamedDecl *>(Friend)) {
    assert(isa<FunctionDecl>(D) ||
           isa<CXXRecordDecl>(D) ||
           isa<FunctionTemplateDecl>(D) ||
           isa<ClassTemplateDecl>(D));

    // As a temporary hack, we permit template instantiation to point
    // to the original declaration when instantiating members.
    assert(D->getFriendObjectKind() ||
           (cast<CXXRecordDecl>(DC)->getTemplateSpecializationKind()));
  }
#endif

  auto *FD =
      new (C, DC) FriendDecl(Decl::Friend, DC, L, Friend, FriendL, EllipsisLoc);
  cast<CXXRecordDecl>(DC)->pushFriendDecl(FD);
  return FD;
}

FriendDecl *FriendDecl::CreateDeserialized(ASTContext &C, GlobalDeclID ID) {
  return new (C, ID) FriendDecl(Decl::Friend, EmptyShell());
}

FriendDecl *FriendDecl::getNextFriendSlowCase() {
  return cast_or_null<FriendDecl>(
      NextFriend.get(getASTContext().getExternalSource()));
}

FriendDecl *CXXRecordDecl::getFirstFriend() const {
  ExternalASTSource *Source = getParentASTContext().getExternalSource();
  Decl *First = data().FirstFriend.get(Source);
  return First ? cast<FriendDecl>(First) : nullptr;
}

SourceRange FriendDecl::getSourceRange() const {
  if (TypeSourceInfo *TInfo = getFriendType()) {
    SourceLocation EndL =
        isPackExpansion() ? getEllipsisLoc() : TInfo->getTypeLoc().getEndLoc();
    return SourceRange(getFriendLoc(), EndL);
  }

  if (isPackExpansion())
    return SourceRange(getFriendLoc(), getEllipsisLoc());

  if (NamedDecl *ND = getFriendDecl()) {
    if (const auto *FD = dyn_cast<FunctionDecl>(ND))
      return FD->getSourceRange();
    if (const auto *FTD = dyn_cast<FunctionTemplateDecl>(ND))
      return FTD->getSourceRange();
    if (const auto *CTD = dyn_cast<ClassTemplateDecl>(ND))
      return CTD->getSourceRange();
    if (const auto *DD = dyn_cast<DeclaratorDecl>(ND)) {
      if (DD->getOuterLocStart() != DD->getInnerLocStart())
        return DD->getSourceRange();
    }
    return SourceRange(getFriendLoc(), ND->getEndLoc());
  }
  return SourceRange(getFriendLoc(), getLocation());
}
