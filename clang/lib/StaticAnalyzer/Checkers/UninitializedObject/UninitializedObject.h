//===----- UninitializedObject.h ---------------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines helper classes for UninitializedObjectChecker and
// documentation about the logic of it.
//
// To read about command line options and a description what this checker does,
// refer to UninitializedObjectChecker.cpp.
//
// Some methods are implemented in UninitializedPointee.cpp, to reduce the
// complexity of the main checker file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_UNINITIALIZEDOBJECT_H
#define LLVM_CLANG_STATICANALYZER_UNINITIALIZEDOBJECT_H

#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

namespace clang {
namespace ento {

/// Represents a field chain. A field chain is a vector of fields where the
/// first element of the chain is the object under checking (not stored), and
/// every other element is a field, and the element that precedes it is the
/// object that contains it.
///
/// Note that this class is immutable, and new fields may only be added through
/// constructor calls.
class FieldChainInfo {
public:
  using FieldChainImpl = llvm::ImmutableListImpl<const FieldRegion *>;
  using FieldChain = llvm::ImmutableList<const FieldRegion *>;

private:
  FieldChain::Factory &Factory;
  FieldChain Chain;

  const bool IsDereferenced = false;

public:
  FieldChainInfo() = delete;
  FieldChainInfo(FieldChain::Factory &F) : Factory(F) {}

  FieldChainInfo(const FieldChainInfo &Other, const bool IsDereferenced)
      : Factory(Other.Factory), Chain(Other.Chain),
        IsDereferenced(IsDereferenced) {}

  FieldChainInfo(const FieldChainInfo &Other, const FieldRegion *FR,
                 const bool IsDereferenced = false);

  bool contains(const FieldRegion *FR) const { return Chain.contains(FR); }
  bool isPointer() const;

  /// If this is a fieldchain whose last element is an uninitialized region of a
  /// pointer type, `IsDereferenced` will store whether the pointer itself or
  /// the pointee is uninitialized.
  bool isDereferenced() const;
  const FieldDecl *getEndOfChain() const;
  void print(llvm::raw_ostream &Out) const;

private:
  friend struct FieldChainInfoComparator;
};

struct FieldChainInfoComparator {
  bool operator()(const FieldChainInfo &lhs, const FieldChainInfo &rhs) const {
    assert(!lhs.Chain.isEmpty() && !rhs.Chain.isEmpty() &&
           "Attempted to store an empty fieldchain!");
    return *lhs.Chain.begin() < *rhs.Chain.begin();
  }
};

using UninitFieldSet = std::set<FieldChainInfo, FieldChainInfoComparator>;

/// Searches for and stores uninitialized fields in a non-union object.
class FindUninitializedFields {
  ProgramStateRef State;
  const TypedValueRegion *const ObjectR;

  const bool IsPedantic;
  const bool CheckPointeeInitialization;

  bool IsAnyFieldInitialized = false;

  FieldChainInfo::FieldChain::Factory Factory;
  UninitFieldSet UninitFields;

public:
  FindUninitializedFields(ProgramStateRef State,
                          const TypedValueRegion *const R, bool IsPedantic,
                          bool CheckPointeeInitialization);
  const UninitFieldSet &getUninitFields();

private:
  /// Adds a FieldChainInfo object to UninitFields. Return true if an insertion
  /// took place.
  bool addFieldToUninits(FieldChainInfo LocalChain);

  // For the purposes of this checker, we'll regard the object under checking as
  // a directed tree, where
  //   * the root is the object under checking
  //   * every node is an object that is
  //     - a union
  //     - a non-union record
  //     - a pointer/reference
  //     - an array
  //     - of a primitive type, which we'll define later in a helper function.
  //   * the parent of each node is the object that contains it
  //   * every leaf is an array, a primitive object, a nullptr or an undefined
  //   pointer.
  //
  // Example:
  //
  //   struct A {
  //      struct B {
  //        int x, y = 0;
  //      };
  //      B b;
  //      int *iptr = new int;
  //      B* bptr;
  //
  //      A() {}
  //   };
  //
  // The directed tree:
  //
  //           ->x
  //          /
  //      ->b--->y
  //     /
  //    A-->iptr->(int value)
  //     \
  //      ->bptr
  //
  // From this we'll construct a vector of fieldchains, where each fieldchain
  // represents an uninitialized field. An uninitialized field may be a
  // primitive object, a pointer, a pointee or a union without a single
  // initialized field.
  // In the above example, for the default constructor call we'll end up with
  // these fieldchains:
  //
  //   this->b.x
  //   this->iptr (pointee uninit)
  //   this->bptr (pointer uninit)
  //
  // We'll traverse each node of the above graph with the appropiate one of
  // these methods:

  /// This method checks a region of a union object, and returns true if no
  /// field is initialized within the region.
  bool isUnionUninit(const TypedValueRegion *R);

  /// This method checks a region of a non-union object, and returns true if
  /// an uninitialized field is found within the region.
  bool isNonUnionUninit(const TypedValueRegion *R, FieldChainInfo LocalChain);

  /// This method checks a region of a pointer or reference object, and returns
  /// true if the ptr/ref object itself or any field within the pointee's region
  /// is uninitialized.
  bool isPointerOrReferenceUninit(const FieldRegion *FR,
                                  FieldChainInfo LocalChain);

  /// This method returns true if the value of a primitive object is
  /// uninitialized.
  bool isPrimitiveUninit(const SVal &V);

  // Note that we don't have a method for arrays -- the elements of an array are
  // often left uninitialized intentionally even when it is of a C++ record
  // type, so we'll assume that an array is always initialized.
  // TODO: Add a support for nonloc::LocAsInteger.
};

/// Returns true if T is a primitive type. We defined this type so that for
/// objects that we'd only like analyze as much as checking whether their
/// value is undefined or not, such as ints and doubles, can be analyzed with
/// ease. This also helps ensuring that every special field type is handled
/// correctly.
static bool isPrimitiveType(const QualType &T) {
  return T->isBuiltinType() || T->isEnumeralType() || T->isMemberPointerType();
}

} // end of namespace ento
} // end of namespace clang

#endif // LLVM_CLANG_STATICANALYZER_UNINITIALIZEDOBJECT_H
