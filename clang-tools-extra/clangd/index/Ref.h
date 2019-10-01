//===--- Ref.h ---------------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_REF_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_REF_H

#include "SymbolID.h"
#include "SymbolLocation.h"
#include "clang/Index/IndexSymbol.h"
#include "llvm/ADT/DenseMap.h"
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
  Declaration = static_cast<uint8_t>(index::SymbolRole::Declaration),
  Definition = static_cast<uint8_t>(index::SymbolRole::Definition),
  Reference = static_cast<uint8_t>(index::SymbolRole::Reference),
  All = Declaration | Definition | Reference,
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
};

inline bool operator<(const Ref &L, const Ref &R) {
  return std::tie(L.Location, L.Kind) < std::tie(R.Location, R.Kind);
}
inline bool operator==(const Ref &L, const Ref &R) {
  return std::tie(L.Location, L.Kind) == std::tie(R.Location, R.Kind);
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
    llvm::BumpPtrAllocator Arena;
    llvm::UniqueStringSaver UniqueStrings; // Contents on the arena.
    llvm::DenseMap<SymbolID, std::set<Ref>> Refs;
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

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_REF_H
