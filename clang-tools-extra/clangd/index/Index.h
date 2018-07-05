//===--- Index.h ------------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_INDEX_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_INDEX_H

#include "clang/Index/IndexSymbol.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringExtras.h"
#include <array>
#include <string>

namespace clang {
namespace clangd {

struct SymbolLocation {
  // Specify a position (Line, Column) of symbol. Using Line/Column allows us to
  // build LSP responses without reading the file content.
  struct Position {
    uint32_t Line = 0; // 0-based
    // Using UTF-16 code units.
    uint32_t Column = 0; // 0-based
  };

  // The URI of the source file where a symbol occurs.
  llvm::StringRef FileURI;

  /// The symbol range, using half-open range [Start, End).
  Position Start;
  Position End;

  operator bool() const { return !FileURI.empty(); }
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const SymbolLocation &);

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
  explicit SymbolID(llvm::StringRef USR);

  bool operator==(const SymbolID &Sym) const {
    return HashValue == Sym.HashValue;
  }
  bool operator<(const SymbolID &Sym) const {
    return HashValue < Sym.HashValue;
  }

  // Returns a 40-bytes hex encoded string.
  std::string str() const;

private:
  static constexpr unsigned HashByteLength = 20;

  friend llvm::hash_code hash_value(const SymbolID &ID) {
    // We already have a good hash, just return the first bytes.
    static_assert(sizeof(size_t) <= HashByteLength, "size_t longer than SHA1!");
    size_t Result;
    memcpy(&Result, ID.HashValue.data(), sizeof(size_t));
    return llvm::hash_code(Result);
  }
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                       const SymbolID &ID);
  friend void operator>>(llvm::StringRef Str, SymbolID &ID);

  std::array<uint8_t, HashByteLength> HashValue;
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

// Describes the source of information about a symbol.
// Mainly useful for debugging, e.g. understanding code completion reuslts.
// This is a bitfield as information can be combined from several sources.
enum SymbolOrigin : uint8_t {
  Unknown = 0,
  AST = 1 << 0,     // Directly from the AST (indexes should not set this).
  Dynamic = 1 << 1, // From the dynamic index of opened files.
  Static = 1 << 2,  // From the static, externally-built index.
  Merge = 1 << 3,   // A non-trivial index merge was performed.
  // Remaining bits reserved for index implementations.
};
raw_ostream &operator<<(raw_ostream &, SymbolOrigin);

// The class presents a C++ symbol, e.g. class, function.
//
// WARNING: Symbols do not own much of their underlying data - typically strings
// are owned by a SymbolSlab. They should be treated as non-owning references.
// Copies are shallow.
// When adding new unowned data fields to Symbol, remember to update:
//   - SymbolSlab::Builder in Index.cpp, to copy them to the slab's storage.
//   - mergeSymbol in Merge.cpp, to properly combine two Symbols.
//
// A fully documented symbol can be split as:
// size_type std::map<k, t>::count(const K& key) const
// | Return  |     Scope     |Name|    Signature     |
// We split up these components to allow display flexibility later.
struct Symbol {
  // The ID of the symbol.
  SymbolID ID;
  // The symbol information, like symbol kind.
  index::SymbolInfo SymInfo;
  // The unqualified name of the symbol, e.g. "bar" (for ns::bar).
  llvm::StringRef Name;
  // The containing namespace. e.g. "" (global), "ns::" (top-level namespace).
  llvm::StringRef Scope;
  // The location of the symbol's definition, if one was found.
  // This just covers the symbol name (e.g. without class/function body).
  SymbolLocation Definition;
  // The location of the preferred declaration of the symbol.
  // This just covers the symbol name.
  // This may be the same as Definition.
  //
  // A C++ symbol may have multiple declarations, and we pick one to prefer.
  //   * For classes, the canonical declaration should be the definition.
  //   * For non-inline functions, the canonical declaration typically appears
  //     in the ".h" file corresponding to the definition.
  SymbolLocation CanonicalDeclaration;
  // The number of translation units that reference this symbol from their main
  // file. This number is only meaningful if aggregated in an index.
  unsigned References = 0;
  /// Whether or not this symbol is meant to be used for the code completion.
  /// See also isIndexedForCodeCompletion().
  bool IsIndexedForCodeCompletion = false;
  /// Where this symbol came from. Usually an index provides a constant value.
  SymbolOrigin Origin = Unknown;
  /// A brief description of the symbol that can be appended in the completion
  /// candidate list. For example, "(X x, Y y) const" is a function signature.
  llvm::StringRef Signature;
  /// What to insert when completing this symbol, after the symbol name.
  /// This is in LSP snippet syntax (e.g. "({$0})" for a no-args function).
  /// (When snippets are disabled, the symbol name alone is used).
  llvm::StringRef CompletionSnippetSuffix;

  /// Optional symbol details that are not required to be set. For example, an
  /// index fuzzy match can return a large number of symbol candidates, and it
  /// is preferable to send only core symbol information in the batched results
  /// and have clients resolve full symbol information for a specific candidate
  /// if needed.
  struct Details {
    /// Documentation including comment for the symbol declaration.
    llvm::StringRef Documentation;
    /// Type when this symbol is used in an expression. (Short display form).
    /// e.g. return type of a function, or type of a variable.
    llvm::StringRef ReturnType;
    /// This can be either a URI of the header to be #include'd for this symbol,
    /// or a literal header quoted with <> or "" that is suitable to be included
    /// directly. When this is a URI, the exact #include path needs to be
    /// calculated according to the URI scheme.
    ///
    /// This is a canonical include for the symbol and can be different from
    /// FileURI in the CanonicalDeclaration.
    llvm::StringRef IncludeHeader;
  };

  // Optional details of the symbol.
  const Details *Detail = nullptr;

  // FIXME: add all occurrences support.
  // FIXME: add extra fields for index scoring signals.
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Symbol &S);

// Computes query-independent quality score for a Symbol.
// This currently falls in the range [1, ln(#indexed documents)].
// FIXME: this should probably be split into symbol -> signals
//        and signals -> score, so it can be reused for Sema completions.
double quality(const Symbol &S);

// An immutable symbol container that stores a set of symbols.
// The container will maintain the lifetime of the symbols.
class SymbolSlab {
public:
  using const_iterator = std::vector<Symbol>::const_iterator;
  using iterator = const_iterator;

  SymbolSlab() = default;

  const_iterator begin() const { return Symbols.begin(); }
  const_iterator end() const { return Symbols.end(); }
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
    void insert(const Symbol &S);

    // Returns the symbol with an ID, if it exists. Valid until next insert().
    const Symbol *find(const SymbolID &ID) {
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
  /// (e.g. namespaces) excluding nested scopes. For example, if a scope "xyz::"
  /// is provided, the matched symbols must be defined in namespace xyz but not
  /// namespace xyz::abc.
  ///
  /// The global scope is "", a top level scope is "foo::", etc.
  std::vector<std::string> Scopes;
  /// \brief The number of top candidates to return. The index may choose to
  /// return more than this, e.g. if it doesn't know which candidates are best.
  size_t MaxCandidateCount = UINT_MAX;
  /// If set to true, only symbols for completion support will be considered.
  bool RestrictForCodeCompletion = false;
  /// Contextually relevant files (e.g. the file we're code-completing in).
  /// Paths should be absolute.
  std::vector<std::string> ProximityPaths;
};

struct LookupRequest {
  llvm::DenseSet<SymbolID> IDs;
};

/// \brief Interface for symbol indexes that can be used for searching or
/// matching symbols among a set of symbols based on names or unique IDs.
class SymbolIndex {
public:
  virtual ~SymbolIndex() = default;

  /// \brief Matches symbols in the index fuzzily and applies \p Callback on
  /// each matched symbol before returning.
  /// If returned Symbols are used outside Callback, they must be deep-copied!
  ///
  /// Returns true if there may be more results (limited by MaxCandidateCount).
  virtual bool
  fuzzyFind(const FuzzyFindRequest &Req,
            llvm::function_ref<void(const Symbol &)> Callback) const = 0;

  /// Looks up symbols with any of the given symbol IDs and applies \p Callback
  /// on each matched symbol.
  /// The returned symbol must be deep-copied if it's used outside Callback.
  virtual void
  lookup(const LookupRequest &Req,
         llvm::function_ref<void(const Symbol &)> Callback) const = 0;

  // FIXME: add interfaces for more index use cases:
  //  - getAllOccurrences(SymbolID);
};

} // namespace clangd
} // namespace clang
#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_INDEX_H
