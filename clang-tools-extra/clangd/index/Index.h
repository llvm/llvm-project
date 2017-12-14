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
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringExtras.h"
#include <array>
#include <string>

namespace clang {
namespace clangd {

struct SymbolLocation {
  // The absolute path of the source file where a symbol occurs.
  std::string FilePath;
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

private:
  friend llvm::hash_code hash_value(const SymbolID &ID) {
    return hash_value(ArrayRef<uint8_t>(ID.HashValue));
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

// The class presents a C++ symbol, e.g. class, function.
//
// FIXME: instead of having own copy fields for each symbol, we can share
// storage from SymbolSlab.
struct Symbol {
  // The ID of the symbol.
  SymbolID ID;
  // The qualified name of the symbol, e.g. Foo::bar.
  std::string QualifiedName;
  // The symbol information, like symbol kind.
  index::SymbolInfo SymInfo;
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

// A symbol container that stores a set of symbols. The container will maintain
// the lifetime of the symbols.
//
// FIXME: Use a space-efficient implementation, a lot of Symbol fields could
// share the same storage.
class SymbolSlab {
public:
  using const_iterator = llvm::DenseMap<SymbolID, Symbol>::const_iterator;

  SymbolSlab() = default;

  const_iterator begin() const;
  const_iterator end() const;
  const_iterator find(const SymbolID &SymID) const;

  // Once called, no more symbols would be added to the SymbolSlab. This
  // operation is irreversible.
  void freeze();

  void insert(Symbol S);

private:
  bool Frozen = false;

  llvm::DenseMap<SymbolID, Symbol> Symbols;
};

struct FuzzyFindRequest {
  /// \brief A query string for the fuzzy find. This is matched against symbols'
  /// qualfified names.
  std::string Query;
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
  fuzzyFind(Context &Ctx, const FuzzyFindRequest &Req,
            std::function<void(const Symbol &)> Callback) const = 0;

  // FIXME: add interfaces for more index use cases:
  //  - Symbol getSymbolInfo(SymbolID);
  //  - getAllOccurrences(SymbolID);
};

} // namespace clangd
} // namespace clang

namespace llvm {

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

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_INDEX_H
