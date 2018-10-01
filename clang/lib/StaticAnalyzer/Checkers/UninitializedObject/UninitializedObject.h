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

/// Represent a single field. This is only an interface to abstract away special
/// cases like pointers/references.
class FieldNode {
protected:
  const FieldRegion *FR;

public:
  FieldNode(const FieldRegion *FR) : FR(FR) { assert(FR); }

  FieldNode() = delete;
  FieldNode(const FieldNode &) = delete;
  FieldNode(FieldNode &&) = delete;
  FieldNode &operator=(const FieldNode &) = delete;
  FieldNode &operator=(const FieldNode &&) = delete;

  /// Profile - Used to profile the contents of this object for inclusion in a
  /// FoldingSet.
  void Profile(llvm::FoldingSetNodeID &ID) const { ID.AddPointer(this); }

  bool operator<(const FieldNode &Other) const { return FR < Other.FR; }
  bool isSameRegion(const FieldRegion *OtherFR) const { return FR == OtherFR; }

  const FieldRegion *getRegion() const { return FR; }
  const FieldDecl *getDecl() const { return FR->getDecl(); }

  // When a fieldchain is printed (a list of FieldNode objects), it will have
  // the following format:
  // <note message>'<prefix>this-><node><separator><node><separator>...<node>'

  /// If this is the last element of the fieldchain, this method will be called.
  /// The note message should state something like "uninitialized field" or
  /// "uninitialized pointee" etc.
  virtual void printNoteMsg(llvm::raw_ostream &Out) const = 0;

  /// Print any prefixes before the fieldchain.
  virtual void printPrefix(llvm::raw_ostream &Out) const = 0;

  /// Print the node. Should contain the name of the field stored in getRegion.
  virtual void printNode(llvm::raw_ostream &Out) const = 0;

  /// Print the separator. For example, fields may be separated with '.' or
  /// "->".
  virtual void printSeparator(llvm::raw_ostream &Out) const = 0;
};

/// Returns with Field's name. This is a helper function to get the correct name
/// even if Field is a captured lambda variable.
StringRef getVariableName(const FieldDecl *Field);

/// Represents a field chain. A field chain is a vector of fields where the
/// first element of the chain is the object under checking (not stored), and
/// every other element is a field, and the element that precedes it is the
/// object that contains it.
///
/// Note that this class is immutable (essentially a wrapper around an
/// ImmutableList), and new elements can only be added by creating new
/// FieldChainInfo objects through add().
class FieldChainInfo {
public:
  using FieldChainImpl = llvm::ImmutableListImpl<const FieldNode &>;
  using FieldChain = llvm::ImmutableList<const FieldNode &>;

private:
  FieldChain::Factory &ChainFactory;
  FieldChain Chain;

public:
  FieldChainInfo() = delete;
  FieldChainInfo(FieldChain::Factory &F) : ChainFactory(F) {}
  FieldChainInfo(const FieldChainInfo &Other) = default;

  template <class FieldNodeT> FieldChainInfo add(const FieldNodeT &FN);

  bool contains(const FieldRegion *FR) const;
  const FieldRegion *getUninitRegion() const;
  void printNoteMsg(llvm::raw_ostream &Out) const;
};

using UninitFieldMap = std::map<const FieldRegion *, llvm::SmallString<50>>;

/// Searches for and stores uninitialized fields in a non-union object.
class FindUninitializedFields {
  ProgramStateRef State;
  const TypedValueRegion *const ObjectR;

  const bool IsPedantic;
  const bool CheckPointeeInitialization;

  bool IsAnyFieldInitialized = false;

  FieldChainInfo::FieldChain::Factory ChainFactory;

  /// A map for assigning uninitialized regions to note messages. For example,
  ///
  ///   struct A {
  ///     int x;
  ///   };
  ///
  ///   A a;
  ///
  /// After analyzing `a`, the map will contain a pair for `a.x`'s region and
  /// the note message "uninitialized field 'this->x'.
  UninitFieldMap UninitFields;

public:
  FindUninitializedFields(ProgramStateRef State,
                          const TypedValueRegion *const R, bool IsPedantic,
                          bool CheckPointeeInitialization);
  const UninitFieldMap &getUninitFields();

private:
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

  /// Processes LocalChain and attempts to insert it into UninitFields. Returns
  /// true on success.
  ///
  /// Since this class analyzes regions with recursion, we'll only store
  /// references to temporary FieldNode objects created on the stack. This means
  /// that after analyzing a leaf of the directed tree described above, the
  /// elements LocalChain references will be destructed, so we can't store it
  /// directly.
  bool addFieldToUninits(FieldChainInfo LocalChain);
};

/// Returns true if T is a primitive type. We defined this type so that for
/// objects that we'd only like analyze as much as checking whether their
/// value is undefined or not, such as ints and doubles, can be analyzed with
/// ease. This also helps ensuring that every special field type is handled
/// correctly.
static bool isPrimitiveType(const QualType &T) {
  return T->isBuiltinType() || T->isEnumeralType() || T->isMemberPointerType();
}

// Template method definitions.

template <class FieldNodeT>
inline FieldChainInfo FieldChainInfo::add(const FieldNodeT &FN) {
  assert(!contains(FN.getRegion()) &&
         "Can't add a field that is already a part of the "
         "fieldchain! Is this a cyclic reference?");

  FieldChainInfo NewChain = *this;
  NewChain.Chain = ChainFactory.add(FN, Chain);
  return NewChain;
}

} // end of namespace ento
} // end of namespace clang

#endif // LLVM_CLANG_STATICANALYZER_UNINITIALIZEDOBJECT_H
