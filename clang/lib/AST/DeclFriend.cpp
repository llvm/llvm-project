//===--- DeclFriend.cpp - C++ Friend Declaration AST Node Implementation --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AST classes related to C++ friend
// declarations.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/DeclTemplate.h"
using namespace clang;

void FriendDecl::anchor() { }

FriendDecl *FriendDecl::getNextFriendSlowCase() {
  return cast_or_null<FriendDecl>(
                           NextFriend.get(getASTContext().getExternalSource()));
}

FriendDecl *FriendDecl::Create(ASTContext &C, DeclContext *DC,
                               SourceLocation L,
                               FriendUnion Friend,
                               SourceLocation FriendL) {
#ifndef NDEBUG
  if (Friend.is<NamedDecl*>()) {
    NamedDecl *D = Friend.get<NamedDecl*>();
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

  FriendDecl *FD = new (C) FriendDecl(DC, L, Friend, FriendL);
  cast<CXXRecordDecl>(DC)->pushFriendDecl(FD);
  return FD;
}

FriendDecl *FriendDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  void *Mem = AllocateDeserializedDecl(C, ID, sizeof(FriendDecl));
  return new (Mem) FriendDecl(EmptyShell());
}
