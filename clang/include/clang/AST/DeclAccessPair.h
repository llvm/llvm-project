//===--- DeclAccessPair.h - A decl bundled with its path access -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the DeclAccessPair class, which provides an
//  efficient representation of a pair of a NamedDecl* and an
//  AccessSpecifier.  Generally the access specifier gives the
//  natural access of a declaration when named in a class, as
//  defined in C++ [class.access.base]p1.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECLACCESSPAIR_H
#define LLVM_CLANG_AST_DECLACCESSPAIR_H

#include "clang/Basic/Specifiers.h"

namespace clang {

class NamedDecl;

/// A POD class for pairing a NamedDecl* with an access specifier.
/// Can be put into unions.
class DeclAccessPair {
  NamedDecl *Ptr; // we'd use llvm::PointerUnion, but it isn't trivial

  enum { Mask = 0x3 };

public:
  static DeclAccessPair make(NamedDecl *D, AccessSpecifier AS) {
    DeclAccessPair p;
    p.set(D, AS);
    return p;
  }

  NamedDecl *getDecl() const {
    return (NamedDecl*) (~Mask & (uintptr_t) Ptr);
  }
  AccessSpecifier getAccess() const {
    return AccessSpecifier(Mask & (uintptr_t) Ptr);
  }

  void setDecl(NamedDecl *D) {
    set(D, getAccess());
  }
  void setAccess(AccessSpecifier AS) {
    set(getDecl(), AS);
  }
  void set(NamedDecl *D, AccessSpecifier AS) {
    Ptr = reinterpret_cast<NamedDecl*>(uintptr_t(AS) |
                                       reinterpret_cast<uintptr_t>(D));
  }

  operator NamedDecl*() const { return getDecl(); }
  NamedDecl *operator->() const { return getDecl(); }
};
}

// Take a moment to tell SmallVector that DeclAccessPair is POD.
namespace llvm {
template<typename> struct isPodLike;
template<> struct isPodLike<clang::DeclAccessPair> {
   static const bool value = true;
};
}

#endif
