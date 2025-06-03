//===--- Ref.h ---------------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_REF_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_REF_H

#include "index/SymbolID.h"
#include "index/SymbolLocation.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <set>
#include <utility>

namespace clang {
namespace clangd {

/// Describes the kind of a cross-reference.
///
/// This is a bitfield which can be combined from different kinds.
enum class RefKind : uint8_t {
  Unknown = 0,
  // Points to symbol declaration. Example:
  //
  // class Foo;
  //       ^ Foo declaration
  // Foo foo;
  // ^ this does not reference Foo declaration
  Declaration = 1 << 0,
  // Points to symbol definition. Example:
  //
  // int foo();
  //     ^ references foo declaration, but not foo definition
  // int foo() { return 42; }
  //     ^ references foo definition, but not declaration
  // bool bar() { return true; }
  //      ^ references both definition and declaration
  Definition = 1 << 1,
  // Points to symbol reference. Example:
  //
  // int Foo = 42;
  // int Bar = Foo + 1;
  //           ^ this is a reference to Foo
  Reference = 1 << 2,
  // The reference explicitly spells out declaration's name. Such references can
  // not come from macro expansions or implicit AST nodes.
  //
  // class Foo { public: Foo() {} };
  //       ^ references declaration, definition and explicitly spells out name
  // #define MACRO Foo
  //     v there is an implicit constructor call here which is not a spelled ref
  // Foo foo;
  // ^ this reference explicitly spells out Foo's name
  // struct Bar {
  //   MACRO Internal;
  //   ^ this references Foo, but does not explicitly spell out its name
  // };
  Spelled = 1 << 3,
  // A reference which is a call. Used as a filter for which references
  // to store in data structures used for computing outgoing calls.
  Call = 1 << 4,
  All = Declaration | Definition | Reference | Spelled,
};

inline RefKind operator|(RefKind L, RefKind R) {
  return static_cast<RefKind>(static_cast<uint8_t>(L) |
                              static_cast<uint8_t>(R));
}
inline RefKind &operator|=(RefKind &L, RefKind R) { return L = L | R; }
inline RefKind operator&(RefKind A, RefKind B) {
  return static_cast<RefKind>(static_cast<uint8_t>(A) &
                              static_cast<uint8_t>(B));
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &, RefKind);

/// Represents a symbol occurrence in the source file.
/// Despite the name, it could be a declaration/definition/reference.
///
/// WARNING: Location does not own the underlying data - Copies are shallow.
struct Ref {
  /// The source location where the symbol is named.
  SymbolLocation Location;
  RefKind Kind = RefKind::Unknown;
  /// The ID of the symbol whose definition contains this reference.
  /// For example, for a reference inside a function body, this would
  /// be that function. For top-level definitions this isNull().
  SymbolID Container;
};

inline bool operator<(const Ref &L, const Ref &R) {
  return std::tie(L.Location, L.Kind, L.Container) <
         std::tie(R.Location, R.Kind, R.Container);
}
inline bool operator==(const Ref &L, const Ref &R) {
  return std::tie(L.Location, L.Kind, L.Container) ==
         std::tie(R.Location, R.Kind, R.Container);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Ref &);

/// An efficient structure of storing large set of symbol references in memory.
/// Filenames are deduplicated.
class RefSlab {
public:
  // Refs are stored in order.
  using value_type = std::pair<SymbolID, llvm::ArrayRef<Ref>>;
  using const_iterator = std::vector<value_type>::const_iterator;
  using iterator = const_iterator;

  RefSlab() = default;
  RefSlab(RefSlab &&Slab) = default;
  RefSlab &operator=(RefSlab &&RHS) = default;

  const_iterator begin() const { return Refs.begin(); }
  const_iterator end() const { return Refs.end(); }
  /// Gets the number of symbols.
  size_t size() const { return Refs.size(); }
  size_t numRefs() const { return NumRefs; }
  bool empty() const { return Refs.empty(); }

  size_t bytes() const {
    return sizeof(*this) + Arena.getTotalMemory() +
           sizeof(value_type) * Refs.capacity();
  }

  /// RefSlab::Builder is a mutable container that can 'freeze' to RefSlab.
  class Builder {
  public:
    Builder() : UniqueStrings(Arena) {}
    /// Adds a ref to the slab. Deep copy: Strings will be owned by the slab.
    void insert(const SymbolID &ID, const Ref &S);
    /// Consumes the builder to finalize the slab.
    RefSlab build() &&;

  private:
    // A ref we're storing with its symbol to consume with build().
    // All strings are interned, so DenseMapInfo can use pointer comparisons.
    struct Entry {
      SymbolID Symbol;
      Ref Reference;
    };
    friend struct llvm::DenseMapInfo<Entry>;

    llvm::BumpPtrAllocator Arena;
    llvm::UniqueStringSaver UniqueStrings; // Contents on the arena.
    llvm::DenseSet<Entry> Entries;
  };

private:
  RefSlab(std::vector<value_type> Refs, llvm::BumpPtrAllocator Arena,
          size_t NumRefs)
      : Arena(std::move(Arena)), Refs(std::move(Refs)), NumRefs(NumRefs) {}

  llvm::BumpPtrAllocator Arena;
  std::vector<value_type> Refs;
  /// Number of all references.
  size_t NumRefs = 0;
};

} // namespace clangd
} // namespace clang

namespace llvm {
template <> struct DenseMapInfo<clang::clangd::RefSlab::Builder::Entry> {
  using Entry = clang::clangd::RefSlab::Builder::Entry;
  static inline Entry getEmptyKey() {
    static Entry E{clang::clangd::SymbolID(""), {}};
    return E;
  }
  static inline Entry getTombstoneKey() {
    static Entry E{clang::clangd::SymbolID("TOMBSTONE"), {}};
    return E;
  }
  static unsigned getHashValue(const Entry &Val) {
    return llvm::hash_combine(
        Val.Symbol, reinterpret_cast<uintptr_t>(Val.Reference.Location.FileURI),
        Val.Reference.Location.Start.rep(), Val.Reference.Location.End.rep());
  }
  static bool isEqual(const Entry &LHS, const Entry &RHS) {
    return std::tie(LHS.Symbol, LHS.Reference.Location.FileURI,
                    LHS.Reference.Kind) ==
               std::tie(RHS.Symbol, RHS.Reference.Location.FileURI,
                        RHS.Reference.Kind) &&
           LHS.Reference.Location.Start == RHS.Reference.Location.Start &&
           LHS.Reference.Location.End == RHS.Reference.Location.End;
  }
};
} // namespace llvm

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_REF_H
