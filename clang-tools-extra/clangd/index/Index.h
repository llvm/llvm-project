//===--- Symbol.h -----------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_INDEX_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_INDEX_H

#include "../Context.h"
#include "clang/Index/IndexSymbol.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringExtras.h"
#include <array>
#include <string>

namespace clang {
namespace clangd {

struct SymbolLocation {
  // The absolute path of the source file where a symbol occurs.
  llvm::StringRef FilePath;
  // The 0-based offset to the first character of the symbol from the beginning
  // of the source file.
  unsigned StartOffset;
  // The 0-based offset to the last character of the symbol from the beginning
  // of the source file.
  unsigned EndOffset;
};

// The class identifies a particular C++ symbol (class, function, method, etc).
//
// As USRs (Unified Symbol Resolution) could be large, especially for functions
// with long type arguments, SymbolID is using 160-bits SHA1(USR) values to
// guarantee the uniqueness of symbols while using a relatively small amount of
// memory (vs storing USRs directly).
//
// SymbolID can be used as key in the symbol indexes to lookup the symbol.
class SymbolID {
public:
  SymbolID() = default;
  SymbolID(llvm::StringRef USR);

  bool operator==(const SymbolID &Sym) const {
    return HashValue == Sym.HashValue;
  }
  bool operator<(const SymbolID &Sym) const {
    return HashValue < Sym.HashValue;
  }

private:
  friend llvm::hash_code hash_value(const SymbolID &ID) {
    // We already have a good hash, just return the first bytes.
    static_assert(sizeof(size_t) <= 20, "size_t longer than SHA1!");
    return *reinterpret_cast<const size_t *>(ID.HashValue.data());
  }
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                       const SymbolID &ID);
  friend void operator>>(llvm::StringRef Str, SymbolID &ID);

  std::array<uint8_t, 20> HashValue;
};

// Write SymbolID into the given stream. SymbolID is encoded as a 40-bytes
// hex string.
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const SymbolID &ID);

// Construct SymbolID from a hex string.
// The HexStr is required to be a 40-bytes hex string, which is encoded from the
// "<<" operator.
void operator>>(llvm::StringRef HexStr, SymbolID &ID);

} // namespace clangd
} // namespace clang
namespace llvm {
// Support SymbolIDs as DenseMap keys.
template <> struct DenseMapInfo<clang::clangd::SymbolID> {
  static inline clang::clangd::SymbolID getEmptyKey() {
    static clang::clangd::SymbolID EmptyKey("EMPTYKEY");
    return EmptyKey;
  }
  static inline clang::clangd::SymbolID getTombstoneKey() {
    static clang::clangd::SymbolID TombstoneKey("TOMBSTONEKEY");
    return TombstoneKey;
  }
  static unsigned getHashValue(const clang::clangd::SymbolID &Sym) {
    return hash_value(Sym);
  }
  static bool isEqual(const clang::clangd::SymbolID &LHS,
                      const clang::clangd::SymbolID &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm
namespace clang {
namespace clangd {

// The class presents a C++ symbol, e.g. class, function.
//
// WARNING: Symbols do not own much of their underlying data - typically strings
// are owned by a SymbolSlab. They should be treated as non-owning references.
// Copies are shallow.
// When adding new unowned data fields to Symbol, remember to update
// SymbolSlab::Builder in Index.cpp to copy them to the slab's storage.
struct Symbol {
  // The ID of the symbol.
  SymbolID ID;
  // The symbol information, like symbol kind.
  index::SymbolInfo SymInfo;
  // The unqualified name of the symbol, e.g. "bar" (for "n1::n2::bar").
  llvm::StringRef Name;
  // The scope (e.g. namespace) of the symbol, e.g. "n1::n2" (for
  // "n1::n2::bar").
  llvm::StringRef Scope;
  // The location of the canonical declaration of the symbol.
  //
  // A C++ symbol could have multiple declarations and one definition (e.g.
  // a function is declared in ".h" file, and is defined in ".cc" file).
  //   * For classes, the canonical declaration is usually definition.
  //   * For non-inline functions, the canonical declaration is a declaration
  //     (not a definition), which is usually declared in ".h" file.
  SymbolLocation CanonicalDeclaration;

  // FIXME: add definition location of the symbol.
  // FIXME: add all occurrences support.
  // FIXME: add extra fields for index scoring signals.
  // FIXME: add code completion information.
};

// An immutable symbol container that stores a set of symbols.
// The container will maintain the lifetime of the symbols.
class SymbolSlab {
public:
  using const_iterator = std::vector<Symbol>::const_iterator;

  SymbolSlab() = default;

  const_iterator begin() const;
  const_iterator end() const;
  const_iterator find(const SymbolID &SymID) const;

  size_t size() const { return Symbols.size(); }
  // Estimates the total memory usage.
  size_t bytes() const {
    return sizeof(*this) + Arena.getTotalMemory() +
           Symbols.capacity() * sizeof(Symbol);
  }

  // SymbolSlab::Builder is a mutable container that can 'freeze' to SymbolSlab.
  // The frozen SymbolSlab will use less memory.
  class Builder {
   public:
     // Adds a symbol, overwriting any existing one with the same ID.
     // This is a deep copy: underlying strings will be owned by the slab.
     void insert(const Symbol& S);

     // Returns the symbol with an ID, if it exists. Valid until next insert().
     const Symbol* find(const SymbolID &ID) {
       auto I = SymbolIndex.find(ID);
       return I == SymbolIndex.end() ? nullptr : &Symbols[I->second];
     }

     // Consumes the builder to finalize the slab.
     SymbolSlab build() &&;

   private:
     llvm::BumpPtrAllocator Arena;
     // Intern table for strings. Contents are on the arena.
     llvm::DenseSet<llvm::StringRef> Strings;
     std::vector<Symbol> Symbols;
     // Values are indices into Symbols vector.
     llvm::DenseMap<SymbolID, size_t> SymbolIndex;
  };

private:
  SymbolSlab(llvm::BumpPtrAllocator Arena, std::vector<Symbol> Symbols)
      : Arena(std::move(Arena)), Symbols(std::move(Symbols)) {}

  llvm::BumpPtrAllocator Arena; // Owns Symbol data that the Symbols do not.
  std::vector<Symbol> Symbols;  // Sorted by SymbolID to allow lookup.
};

struct FuzzyFindRequest {
  /// \brief A query string for the fuzzy find. This is matched against symbols'
  /// un-qualified identifiers and should not contain qualifiers like "::".
  std::string Query;
  /// \brief If this is non-empty, symbols must be in at least one of the scopes
  /// (e.g. namespaces) excluding nested scopes. For example, if a scope "xyz"
  /// is provided, the matched symbols must be defined in scope "xyz" but not
  /// "xyz::abc".
  ///
  /// A scope must be fully qualified without leading or trailing "::" e.g.
  /// "n1::n2". "" is interpreted as the global namespace, and "::" is invalid.
  std::vector<std::string> Scopes;
  /// \brief The maxinum number of candidates to return.
  size_t MaxCandidateCount = UINT_MAX;
};

/// \brief Interface for symbol indexes that can be used for searching or
/// matching symbols among a set of symbols based on names or unique IDs.
class SymbolIndex {
public:
  virtual ~SymbolIndex() = default;

  /// \brief Matches symbols in the index fuzzily and applies \p Callback on
  /// each matched symbol before returning.
  ///
  /// Returns true if the result list is complete, false if it was truncated due
  /// to MaxCandidateCount
  virtual bool
  fuzzyFind(const Context &Ctx, const FuzzyFindRequest &Req,
            std::function<void(const Symbol &)> Callback) const = 0;

  // FIXME: add interfaces for more index use cases:
  //  - Symbol getSymbolInfo(SymbolID);
  //  - getAllOccurrences(SymbolID);
};

} // namespace clangd
} // namespace clang
#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_INDEX_H
