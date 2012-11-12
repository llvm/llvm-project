//===-- Redeclarable.h - Base for Decls that can be redeclared -*- C++ -*-====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Redeclarable interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_REDECLARABLE_H
#define LLVM_CLANG_AST_REDECLARABLE_H

#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/Casting.h"
#include <iterator>

namespace clang {

/// \brief Provides common interface for the Decls that can be redeclared.
template<typename decl_type>
class Redeclarable {

protected:
  class DeclLink {
    llvm::PointerIntPair<decl_type *, 1, bool> NextAndIsPrevious;
  public:
    DeclLink(decl_type *D, bool isLatest)
      : NextAndIsPrevious(D, isLatest) { }

    bool NextIsPrevious() const { return !NextAndIsPrevious.getInt(); }
    bool NextIsLatest() const { return NextAndIsPrevious.getInt(); }
    decl_type *getNext() const { return NextAndIsPrevious.getPointer(); }
    void setNext(decl_type *D) { NextAndIsPrevious.setPointer(D); }
  };

  static DeclLink PreviousDeclLink(decl_type *D) {
    return DeclLink(D, false);
  }

  static DeclLink LatestDeclLink(decl_type *D) {
    return DeclLink(D, true);
  }

  /// \brief Points to the next redeclaration in the chain.
  ///
  /// If NextIsPrevious() is true, this is a link to the previous declaration
  /// of this same Decl. If NextIsLatest() is true, this is the first
  /// declaration and Link points to the latest declaration. For example:
  ///
  ///  #1 int f(int x, int y = 1); // <pointer to #3, true>
  ///  #2 int f(int x = 0, int y); // <pointer to #1, false>
  ///  #3 int f(int x, int y) { return x + y; } // <pointer to #2, false>
  ///
  /// If there is only one declaration, it is <pointer to self, true>
  DeclLink RedeclLink;

public:
  Redeclarable() : RedeclLink(LatestDeclLink(static_cast<decl_type*>(this))) { }

  /// \brief Return the previous declaration of this declaration or NULL if this
  /// is the first declaration.
  decl_type *getPreviousDecl() {
    if (RedeclLink.NextIsPrevious())
      return RedeclLink.getNext();
    return 0;
  }
  const decl_type *getPreviousDecl() const {
    return const_cast<decl_type *>(
                 static_cast<const decl_type*>(this))->getPreviousDecl();
  }

  /// \brief Return the first declaration of this declaration or itself if this
  /// is the only declaration.
  decl_type *getFirstDeclaration() {
    decl_type *D = static_cast<decl_type*>(this);
    while (D->getPreviousDecl())
      D = D->getPreviousDecl();
    return D;
  }

  /// \brief Return the first declaration of this declaration or itself if this
  /// is the only declaration.
  const decl_type *getFirstDeclaration() const {
    const decl_type *D = static_cast<const decl_type*>(this);
    while (D->getPreviousDecl())
      D = D->getPreviousDecl();
    return D;
  }

  /// \brief Returns true if this is the first declaration.
  bool isFirstDeclaration() const {
    return RedeclLink.NextIsLatest();
  }

  /// \brief Returns the most recent (re)declaration of this declaration.
  decl_type *getMostRecentDecl() {
    return getFirstDeclaration()->RedeclLink.getNext();
  }

  /// \brief Returns the most recent (re)declaration of this declaration.
  const decl_type *getMostRecentDecl() const {
    return getFirstDeclaration()->RedeclLink.getNext();
  }
  
  /// \brief Set the previous declaration. If PrevDecl is NULL, set this as the
  /// first and only declaration.
  void setPreviousDeclaration(decl_type *PrevDecl);

  /// \brief Iterates through all the redeclarations of the same decl.
  class redecl_iterator {
    /// Current - The current declaration.
    decl_type *Current;
    decl_type *Starter;
    bool PassedFirst;

  public:
    typedef decl_type*                value_type;
    typedef decl_type*                reference;
    typedef decl_type*                pointer;
    typedef std::forward_iterator_tag iterator_category;
    typedef std::ptrdiff_t            difference_type;

    redecl_iterator() : Current(0) { }
    explicit redecl_iterator(decl_type *C)
      : Current(C), Starter(C), PassedFirst(false) { }

    reference operator*() const { return Current; }
    pointer operator->() const { return Current; }

    redecl_iterator& operator++() {
      assert(Current && "Advancing while iterator has reached end");
      // Sanity check to avoid infinite loop on invalid redecl chain.
      if (Current->isFirstDeclaration()) {
        if (PassedFirst) {
          assert(0 && "Passed first decl twice, invalid redecl chain!");
          Current = 0;
          return *this;
        }
        PassedFirst = true;
      }

      // Get either previous decl or latest decl.
      decl_type *Next = Current->RedeclLink.getNext();
      Current = (Next != Starter ? Next : 0);
      return *this;
    }

    redecl_iterator operator++(int) {
      redecl_iterator tmp(*this);
      ++(*this);
      return tmp;
    }

    friend bool operator==(redecl_iterator x, redecl_iterator y) {
      return x.Current == y.Current;
    }
    friend bool operator!=(redecl_iterator x, redecl_iterator y) {
      return x.Current != y.Current;
    }
  };

  /// \brief Returns iterator for all the redeclarations of the same decl.
  /// It will iterate at least once (when this decl is the only one).
  redecl_iterator redecls_begin() const {
    return redecl_iterator(const_cast<decl_type*>(
                                          static_cast<const decl_type*>(this)));
  }
  redecl_iterator redecls_end() const { return redecl_iterator(); }

  friend class ASTDeclReader;
  friend class ASTDeclWriter;
};

}

#endif
