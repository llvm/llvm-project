//===- DeclFriend.h - Classes for C++ friend declarations -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the section of the AST representing C++ friend
// declarations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECLFRIEND_H
#define LLVM_CLANG_AST_DECLFRIEND_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include <cassert>

namespace clang {

class ASTContext;

/// FriendDecl - Represents the declaration of a friend entity,
/// which can be a function, a type, or a templated function or type.
/// For example:
///
/// @code
/// template <typename T> class A {
///   friend int foo(T);
///   friend class B;
///   friend T; // only in C++0x
///   template <typename U> friend class C;
///   template <typename U> friend A& operator+=(A&, const U&) { ... }
/// };
/// @endcode
///
/// The semantic context of a friend decl is its declaring class.
class FriendDecl : public Decl {
  LLVM_DECLARE_VIRTUAL_ANCHOR_FUNCTION();

public:
  using FriendUnion = llvm::PointerUnion<NamedDecl *, TypeSourceInfo *>;

private:
  friend class CXXRecordDecl;
  friend class CXXRecordDecl::friend_iterator;

  // Location of the '...', if present.
  SourceLocation EllipsisLoc;

  SourceLocation FriendLoc;

protected:
  // The declaration that's a friend of this class.
  FriendUnion Friend;

  LazyDeclPtr NextFriend;

  FriendDecl(Kind K, DeclContext *DC, SourceLocation L, FriendUnion Friend,
             SourceLocation FriendL, SourceLocation EllipsisLoc = {})
      : Decl(K, DC, L), EllipsisLoc(EllipsisLoc), FriendLoc(FriendL),
        Friend(Friend), NextFriend() {}

  FriendDecl(Kind K, EmptyShell Empty) : Decl(K, Empty) {}

  FriendDecl *getNextFriend() {
    if (NextFriend.isOffset())
      return getNextFriendSlowCase();
    return cast_or_null<FriendDecl>(NextFriend.get(nullptr));
  }

  FriendDecl *getNextFriendSlowCase();

public:
  friend class ASTDeclReader;
  friend class ASTDeclWriter;
  friend class ASTNodeImporter;

  static FriendDecl *Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                            FriendUnion Friend_, SourceLocation FriendL,
                            SourceLocation EllipsisLoc = {});
  static FriendDecl *CreateDeserialized(ASTContext &C, GlobalDeclID ID);

  /// If this friend declaration names an (untemplated but possibly
  /// dependent) type, return the type; otherwise return null.  This
  /// is used for elaborated-type-specifiers and, in C++0x, for
  /// arbitrary friend type declarations.
  TypeSourceInfo *getFriendType() const {
    return Friend.dyn_cast<TypeSourceInfo*>();
  }

  /// If this friend declaration doesn't name a type, return the inner
  /// declaration.
  NamedDecl *getFriendDecl() const {
    return Friend.dyn_cast<NamedDecl *>();
  }

  /// Retrieves the location of the '...', if present.
  SourceLocation getEllipsisLoc() const { return EllipsisLoc; }

  SourceLocation getFriendLoc() const { return FriendLoc; }

  SourceRange getSourceRange() const override LLVM_READONLY;

  bool isPackExpansion() const { return EllipsisLoc.isValid(); }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) {
    return K >= firstFriend && K <= lastFriend;
  }
};
/// An iterator over the friend declarations of a class.
class CXXRecordDecl::friend_iterator {
  friend class CXXRecordDecl;

  FriendDecl *Ptr;

  explicit friend_iterator(FriendDecl *Ptr) : Ptr(Ptr) {}

public:
  friend_iterator() = default;

  using value_type = FriendDecl *;
  using reference = FriendDecl *;
  using pointer = FriendDecl *;
  using difference_type = int;
  using iterator_category = std::forward_iterator_tag;

  reference operator*() const { return Ptr; }

  friend_iterator &operator++() {
    assert(Ptr && "attempt to increment past end of friend list");
    Ptr = Ptr->getNextFriend();
    return *this;
  }

  friend_iterator operator++(int) {
    friend_iterator tmp = *this;
    ++*this;
    return tmp;
  }

  bool operator==(const friend_iterator &Other) const {
    return Ptr == Other.Ptr;
  }

  bool operator!=(const friend_iterator &Other) const {
    return Ptr != Other.Ptr;
  }

  friend_iterator &operator+=(difference_type N) {
    assert(N >= 0 && "cannot rewind a CXXRecordDecl::friend_iterator");
    while (N--)
      ++*this;
    return *this;
  }

  friend_iterator operator+(difference_type N) const {
    friend_iterator tmp = *this;
    tmp += N;
    return tmp;
  }
};

inline CXXRecordDecl::friend_iterator CXXRecordDecl::friend_begin() const {
  return friend_iterator(getFirstFriend());
}

inline CXXRecordDecl::friend_iterator CXXRecordDecl::friend_end() const {
  return friend_iterator(nullptr);
}

inline CXXRecordDecl::friend_range CXXRecordDecl::friends() const {
  return friend_range(friend_begin(), friend_end());
}

inline void CXXRecordDecl::pushFriendDecl(FriendDecl *FD) {
  assert(!FD->NextFriend && "friend already has next friend?");
  FD->NextFriend = data().FirstFriend;
  data().FirstFriend = FD;
}

} // namespace clang

#endif // LLVM_CLANG_AST_DECLFRIEND_H
