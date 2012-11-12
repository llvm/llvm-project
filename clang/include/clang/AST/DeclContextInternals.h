//===-- DeclContextInternals.h - DeclContext Representation -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the data structures used in the implementation
//  of DeclContext.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_AST_DECLCONTEXTINTERNALS_H
#define LLVM_CLANG_AST_DECLCONTEXTINTERNALS_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/DeclCXX.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>

namespace clang {

class DependentDiagnostic;

/// StoredDeclsList - This is an array of decls optimized a common case of only
/// containing one entry.
struct StoredDeclsList {

  /// DeclsTy - When in vector form, this is what the Data pointer points to.
  typedef SmallVector<NamedDecl *, 4> DeclsTy;

  /// \brief The stored data, which will be either a pointer to a NamedDecl,
  /// or a pointer to a vector.
  llvm::PointerUnion<NamedDecl *, DeclsTy *> Data;

public:
  StoredDeclsList() {}

  StoredDeclsList(const StoredDeclsList &RHS) : Data(RHS.Data) {
    if (DeclsTy *RHSVec = RHS.getAsVector())
      Data = new DeclsTy(*RHSVec);
  }

  ~StoredDeclsList() {
    // If this is a vector-form, free the vector.
    if (DeclsTy *Vector = getAsVector())
      delete Vector;
  }

  StoredDeclsList &operator=(const StoredDeclsList &RHS) {
    if (DeclsTy *Vector = getAsVector())
      delete Vector;
    Data = RHS.Data;
    if (DeclsTy *RHSVec = RHS.getAsVector())
      Data = new DeclsTy(*RHSVec);
    return *this;
  }

  bool isNull() const { return Data.isNull(); }

  NamedDecl *getAsDecl() const {
    return Data.dyn_cast<NamedDecl *>();
  }

  DeclsTy *getAsVector() const {
    return Data.dyn_cast<DeclsTy *>();
  }

  void setOnlyValue(NamedDecl *ND) {
    assert(!getAsVector() && "Not inline");
    Data = ND;
    // Make sure that Data is a plain NamedDecl* so we can use its address
    // at getLookupResult.
    assert(*(NamedDecl **)&Data == ND &&
           "PointerUnion mangles the NamedDecl pointer!");
  }

  void remove(NamedDecl *D) {
    assert(!isNull() && "removing from empty list");
    if (NamedDecl *Singleton = getAsDecl()) {
      assert(Singleton == D && "list is different singleton");
      (void)Singleton;
      Data = (NamedDecl *)0;
      return;
    }

    DeclsTy &Vec = *getAsVector();
    DeclsTy::iterator I = std::find(Vec.begin(), Vec.end(), D);
    assert(I != Vec.end() && "list does not contain decl");
    Vec.erase(I);

    assert(std::find(Vec.begin(), Vec.end(), D)
             == Vec.end() && "list still contains decl");
  }

  /// getLookupResult - Return an array of all the decls that this list
  /// represents.
  DeclContext::lookup_result getLookupResult() {
    if (isNull())
      return DeclContext::lookup_result(DeclContext::lookup_iterator(0),
                                        DeclContext::lookup_iterator(0));

    // If we have a single NamedDecl, return it.
    if (getAsDecl()) {
      assert(!isNull() && "Empty list isn't allowed");

      // Data is a raw pointer to a NamedDecl*, return it.
      void *Ptr = &Data;
      return DeclContext::lookup_result((NamedDecl**)Ptr, (NamedDecl**)Ptr+1);
    }

    assert(getAsVector() && "Must have a vector at this point");
    DeclsTy &Vector = *getAsVector();

    // Otherwise, we have a range result.
    return DeclContext::lookup_result(&Vector[0], &Vector[0]+Vector.size());
  }

  /// HandleRedeclaration - If this is a redeclaration of an existing decl,
  /// replace the old one with D and return true.  Otherwise return false.
  bool HandleRedeclaration(NamedDecl *D) {
    // Most decls only have one entry in their list, special case it.
    if (NamedDecl *OldD = getAsDecl()) {
      if (!D->declarationReplaces(OldD))
        return false;
      setOnlyValue(D);
      return true;
    }

    // Determine if this declaration is actually a redeclaration.
    DeclsTy &Vec = *getAsVector();
    for (DeclsTy::iterator OD = Vec.begin(), ODEnd = Vec.end();
         OD != ODEnd; ++OD) {
      NamedDecl *OldD = *OD;
      if (D->declarationReplaces(OldD)) {
        *OD = D;
        return true;
      }
    }

    return false;
  }

  /// AddSubsequentDecl - This is called on the second and later decl when it is
  /// not a redeclaration to merge it into the appropriate place in our list.
  ///
  void AddSubsequentDecl(NamedDecl *D) {
    // If this is the second decl added to the list, convert this to vector
    // form.
    if (NamedDecl *OldD = getAsDecl()) {
      DeclsTy *VT = new DeclsTy();
      VT->push_back(OldD);
      Data = VT;
    }

    DeclsTy &Vec = *getAsVector();

    // Using directives end up in a special entry which contains only
    // other using directives, so all this logic is wasted for them.
    // But avoiding the logic wastes time in the far-more-common case
    // that we're *not* adding a new using directive.

    // Tag declarations always go at the end of the list so that an
    // iterator which points at the first tag will start a span of
    // decls that only contains tags.
    if (D->hasTagIdentifierNamespace())
      Vec.push_back(D);

    // Resolved using declarations go at the front of the list so that
    // they won't show up in other lookup results.  Unresolved using
    // declarations (which are always in IDNS_Using | IDNS_Ordinary)
    // follow that so that the using declarations will be contiguous.
    else if (D->getIdentifierNamespace() & Decl::IDNS_Using) {
      DeclsTy::iterator I = Vec.begin();
      if (D->getIdentifierNamespace() != Decl::IDNS_Using) {
        while (I != Vec.end() &&
               (*I)->getIdentifierNamespace() == Decl::IDNS_Using)
          ++I;
      }
      Vec.insert(I, D);

    // All other declarations go at the end of the list, but before any
    // tag declarations.  But we can be clever about tag declarations
    // because there can only ever be one in a scope.
    } else if (Vec.back()->hasTagIdentifierNamespace()) {
      NamedDecl *TagD = Vec.back();
      Vec.back() = D;
      Vec.push_back(TagD);
    } else
      Vec.push_back(D);
  }
};

class StoredDeclsMap
  : public llvm::SmallDenseMap<DeclarationName, StoredDeclsList, 4> {

public:
  static void DestroyAll(StoredDeclsMap *Map, bool Dependent);

private:
  friend class ASTContext; // walks the chain deleting these
  friend class DeclContext;
  llvm::PointerIntPair<StoredDeclsMap*, 1> Previous;
};

class DependentStoredDeclsMap : public StoredDeclsMap {
public:
  DependentStoredDeclsMap() : FirstDiagnostic(0) {}

private:
  friend class DependentDiagnostic;
  friend class DeclContext; // iterates over diagnostics

  DependentDiagnostic *FirstDiagnostic;
};

} // end namespace clang

#endif
