//===--- Index.h -------------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_INDEX_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_INDEX_H

#include "ExpectedTypes.h"
#include "SymbolID.h"
#include "clang/Index/IndexSymbol.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/StringSaver.h"
#include <array>
#include <limits>
#include <mutex>
#include <string>
#include <tuple>

namespace clang {
namespace clangd {

struct SymbolLocation {
  // Specify a position (Line, Column) of symbol. Using Line/Column allows us to
  // build LSP responses without reading the file content.
  //
  // Position is encoded into 32 bits to save space.
  // If Line/Column overflow, the value will be their maximum value.
  struct Position {
    Position() : Line(0), Column(0) {}
    void setLine(uint32_t Line);
    uint32_t line() const { return Line; }
    void setColumn(uint32_t Column);
    uint32_t column() const { return Column; }

    bool hasOverflow() const {
      return Line >= MaxLine || Column >= MaxColumn;
    }

    static constexpr uint32_t MaxLine = (1 << 20) - 1;
    static constexpr uint32_t MaxColumn = (1 << 12) - 1;

  private:
    uint32_t Line : 20; // 0-based
    // Using UTF-16 code units.
    uint32_t Column : 12; // 0-based
  };

  /// The symbol range, using half-open range [Start, End).
  Position Start;
  Position End;

  explicit operator bool() const { return !StringRef(FileURI).empty(); }

  // The URI of the source file where a symbol occurs.
  // The string must be null-terminated.
  //
  // We avoid using llvm::StringRef here to save memory.
  // WARNING: unless you know what you are doing, it is recommended to use it
  // via llvm::StringRef.
  const char *FileURI = "";
};
inline bool operator==(const SymbolLocation::Position &L,
                       const SymbolLocation::Position &R) {
  return std::make_tuple(L.line(), L.column()) ==
         std::make_tuple(R.line(), R.column());
}
inline bool operator<(const SymbolLocation::Position &L,
                      const SymbolLocation::Position &R) {
  return std::make_tuple(L.line(), L.column()) <
         std::make_tuple(R.line(), R.column());
}
inline bool operator==(const SymbolLocation &L, const SymbolLocation &R) {
  assert(L.FileURI && R.FileURI);
  return !std::strcmp(L.FileURI, R.FileURI) &&
         std::tie(L.Start, L.End) == std::tie(R.Start, R.End);
}
inline bool operator<(const SymbolLocation &L, const SymbolLocation &R) {
  assert(L.FileURI && R.FileURI);
  int Cmp = std::strcmp(L.FileURI, R.FileURI);
  if (Cmp != 0)
    return Cmp < 0;
  return std::tie(L.Start, L.End) < std::tie(R.Start, R.End);
}
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const SymbolLocation &);

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
enum class SymbolOrigin : uint8_t {
  Unknown = 0,
  AST = 1 << 0,     // Directly from the AST (indexes should not set this).
  Dynamic = 1 << 1, // From the dynamic index of opened files.
  Static = 1 << 2,  // From the static, externally-built index.
  Merge = 1 << 3,   // A non-trivial index merge was performed.
  // Remaining bits reserved for index implementations.
};
inline SymbolOrigin operator|(SymbolOrigin A, SymbolOrigin B) {
  return static_cast<SymbolOrigin>(static_cast<uint8_t>(A) |
                                   static_cast<uint8_t>(B));
}
inline SymbolOrigin &operator|=(SymbolOrigin &A, SymbolOrigin B) {
  return A = A | B;
}
inline SymbolOrigin operator&(SymbolOrigin A, SymbolOrigin B) {
  return static_cast<SymbolOrigin>(static_cast<uint8_t>(A) &
                                   static_cast<uint8_t>(B));
}
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
  /// Where this symbol came from. Usually an index provides a constant value.
  SymbolOrigin Origin = SymbolOrigin::Unknown;
  /// A brief description of the symbol that can be appended in the completion
  /// candidate list. For example, "(X x, Y y) const" is a function signature.
  /// Only set when the symbol is indexed for completion.
  llvm::StringRef Signature;
  /// What to insert when completing this symbol, after the symbol name.
  /// This is in LSP snippet syntax (e.g. "({$0})" for a no-args function).
  /// (When snippets are disabled, the symbol name alone is used).
  /// Only set when the symbol is indexed for completion.
  llvm::StringRef CompletionSnippetSuffix;
  /// Documentation including comment for the symbol declaration.
  llvm::StringRef Documentation;
  /// Type when this symbol is used in an expression. (Short display form).
  /// e.g. return type of a function, or type of a variable.
  /// Only set when the symbol is indexed for completion.
  llvm::StringRef ReturnType;

  /// Raw representation of the OpaqueType of the symbol, used for scoring
  /// purposes.
  /// Only set when the symbol is indexed for completion.
  llvm::StringRef Type;

  struct IncludeHeaderWithReferences {
    IncludeHeaderWithReferences() = default;

    IncludeHeaderWithReferences(llvm::StringRef IncludeHeader,
                                unsigned References)
        : IncludeHeader(IncludeHeader), References(References) {}

    /// This can be either a URI of the header to be #include'd
    /// for this symbol, or a literal header quoted with <> or "" that is
    /// suitable to be included directly. When it is a URI, the exact #include
    /// path needs to be calculated according to the URI scheme.
    ///
    /// Note that the include header is a canonical include for the symbol and
    /// can be different from FileURI in the CanonicalDeclaration.
    llvm::StringRef IncludeHeader = "";
    /// The number of translation units that reference this symbol and include
    /// this header. This number is only meaningful if aggregated in an index.
    unsigned References = 0;
  };
  /// One Symbol can potentially be incuded via different headers.
  ///   - If we haven't seen a definition, this covers all declarations.
  ///   - If we have seen a definition, this covers declarations visible from
  ///   any definition.
  /// Only set when the symbol is indexed for completion.
  llvm::SmallVector<IncludeHeaderWithReferences, 1> IncludeHeaders;

  enum SymbolFlag : uint8_t {
    None = 0,
    /// Whether or not this symbol is meant to be used for the code completion.
    /// See also isIndexedForCodeCompletion().
    /// Note that we don't store completion information (signature, snippet,
    /// type, inclues) if the symbol is not indexed for code completion.
    IndexedForCodeCompletion = 1 << 0,
    /// Indicates if the symbol is deprecated.
    Deprecated = 1 << 1,
    // Symbol is an implementation detail.
    ImplementationDetail = 1 << 2,
    // Symbol is visible to other files (not e.g. a static helper function).
    VisibleOutsideFile = 1 << 3,
  };

  SymbolFlag Flags = SymbolFlag::None;
  /// FIXME: also add deprecation message and fixit?
};
inline Symbol::SymbolFlag  operator|(Symbol::SymbolFlag A, Symbol::SymbolFlag  B) {
  return static_cast<Symbol::SymbolFlag>(static_cast<uint8_t>(A) |
                                         static_cast<uint8_t>(B));
}
inline Symbol::SymbolFlag &operator|=(Symbol::SymbolFlag &A, Symbol::SymbolFlag B) {
  return A = A | B;
}
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Symbol &S);
raw_ostream &operator<<(raw_ostream &, Symbol::SymbolFlag);

// Invokes Callback with each StringRef& contained in the Symbol.
// Useful for deduplicating backing strings.
template <typename Callback> void visitStrings(Symbol &S, const Callback &CB) {
  CB(S.Name);
  CB(S.Scope);
  CB(S.Signature);
  CB(S.CompletionSnippetSuffix);
  CB(S.Documentation);
  CB(S.ReturnType);
  CB(S.Type);
  auto RawCharPointerCB = [&CB](const char *&P) {
    llvm::StringRef S(P);
    CB(S);
    assert(!S.data()[S.size()] && "Visited StringRef must be null-terminated");
    P = S.data();
  };
  RawCharPointerCB(S.CanonicalDeclaration.FileURI);
  RawCharPointerCB(S.Definition.FileURI);

  for (auto &Include : S.IncludeHeaders)
    CB(Include.IncludeHeader);
}

// Computes query-independent quality score for a Symbol.
// This currently falls in the range [1, ln(#indexed documents)].
// FIXME: this should probably be split into symbol -> signals
//        and signals -> score, so it can be reused for Sema completions.
float quality(const Symbol &S);

// An immutable symbol container that stores a set of symbols.
// The container will maintain the lifetime of the symbols.
class SymbolSlab {
public:
  using const_iterator = std::vector<Symbol>::const_iterator;
  using iterator = const_iterator;
  using value_type = Symbol;

  SymbolSlab() = default;

  const_iterator begin() const { return Symbols.begin(); }
  const_iterator end() const { return Symbols.end(); }
  const_iterator find(const SymbolID &SymID) const;

  size_t size() const { return Symbols.size(); }
  bool empty() const { return Symbols.empty(); }
  // Estimates the total memory usage.
  size_t bytes() const {
    return sizeof(*this) + Arena.getTotalMemory() +
           Symbols.capacity() * sizeof(Symbol);
  }

  // SymbolSlab::Builder is a mutable container that can 'freeze' to SymbolSlab.
  // The frozen SymbolSlab will use less memory.
  class Builder {
  public:
    Builder() : UniqueStrings(Arena) {}

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
    llvm::UniqueStringSaver UniqueStrings;
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

// Describes the kind of a cross-reference.
//
// This is a bitfield which can be combined from different kinds.
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

// Represents a symbol occurrence in the source file.
// Despite the name, it could be a declaration/definition/reference.
//
// WARNING: Location does not own the underlying data - Copies are shallow.
struct Ref {
  // The source location where the symbol is named.
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

// An efficient structure of storing large set of symbol references in memory.
// Filenames are deduplicated.
class RefSlab {
public:
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
           sizeof(value_type) * Refs.size();
  }

  // RefSlab::Builder is a mutable container that can 'freeze' to RefSlab.
  class Builder {
  public:
    Builder() : UniqueStrings(Arena) {}
    // Adds a ref to the slab. Deep copy: Strings will be owned by the slab.
    void insert(const SymbolID &ID, const Ref &S);
    // Consumes the builder to finalize the slab.
    RefSlab build() &&;

  private:
    llvm::BumpPtrAllocator Arena;
    llvm::UniqueStringSaver UniqueStrings; // Contents on the arena.
    llvm::DenseMap<SymbolID, std::vector<Ref>> Refs;
  };

private:
  RefSlab(std::vector<value_type> Refs, llvm::BumpPtrAllocator Arena,
          size_t NumRefs)
      : Arena(std::move(Arena)), Refs(std::move(Refs)), NumRefs(NumRefs) {}

  llvm::BumpPtrAllocator Arena;
  std::vector<value_type> Refs;
  // Number of all references.
  size_t NumRefs = 0;
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
  /// If set to true, allow symbols from any scope. Scopes explicitly listed
  /// above will be ranked higher.
  bool AnyScope = false;
  /// \brief The number of top candidates to return. The index may choose to
  /// return more than this, e.g. if it doesn't know which candidates are best.
  llvm::Optional<uint32_t> Limit;
  /// If set to true, only symbols for completion support will be considered.
  bool RestrictForCodeCompletion = false;
  /// Contextually relevant files (e.g. the file we're code-completing in).
  /// Paths should be absolute.
  std::vector<std::string> ProximityPaths;

  // FIXME(ibiryukov): add expected type to the request.

  bool operator==(const FuzzyFindRequest &Req) const {
    return std::tie(Query, Scopes, Limit, RestrictForCodeCompletion,
                    ProximityPaths) ==
           std::tie(Req.Query, Req.Scopes, Req.Limit,
                    Req.RestrictForCodeCompletion, Req.ProximityPaths);
  }
  bool operator!=(const FuzzyFindRequest &Req) const { return !(*this == Req); }
};
bool fromJSON(const llvm::json::Value &Value, FuzzyFindRequest &Request);
llvm::json::Value toJSON(const FuzzyFindRequest &Request);

struct LookupRequest {
  llvm::DenseSet<SymbolID> IDs;
};

struct RefsRequest {
  llvm::DenseSet<SymbolID> IDs;
  RefKind Filter = RefKind::All;
  /// If set, limit the number of refers returned from the index. The index may
  /// choose to return less than this, e.g. it tries to avoid returning stale
  /// results.
  llvm::Optional<uint32_t> Limit;
};

/// Interface for symbol indexes that can be used for searching or
/// matching symbols among a set of symbols based on names or unique IDs.
class SymbolIndex {
public:
  virtual ~SymbolIndex() = default;

  /// \brief Matches symbols in the index fuzzily and applies \p Callback on
  /// each matched symbol before returning.
  /// If returned Symbols are used outside Callback, they must be deep-copied!
  ///
  /// Returns true if there may be more results (limited by Req.Limit).
  virtual bool
  fuzzyFind(const FuzzyFindRequest &Req,
            llvm::function_ref<void(const Symbol &)> Callback) const = 0;

  /// Looks up symbols with any of the given symbol IDs and applies \p Callback
  /// on each matched symbol.
  /// The returned symbol must be deep-copied if it's used outside Callback.
  virtual void
  lookup(const LookupRequest &Req,
         llvm::function_ref<void(const Symbol &)> Callback) const = 0;

  /// Finds all occurrences (e.g. references, declarations, definitions) of a
  /// symbol and applies \p Callback on each result.
  ///
  /// Results should be returned in arbitrary order.
  /// The returned result must be deep-copied if it's used outside Callback.
  virtual void refs(const RefsRequest &Req,
                    llvm::function_ref<void(const Ref &)> Callback) const = 0;

  /// Returns estimated size of index (in bytes).
  // FIXME(kbobyrev): Currently, this only returns the size of index itself
  // excluding the size of actual symbol slab index refers to. We should include
  // both.
  virtual size_t estimateMemoryUsage() const = 0;
};

// Delegating implementation of SymbolIndex whose delegate can be swapped out.
class SwapIndex : public SymbolIndex {
public:
  // If an index is not provided, reset() must be called.
  SwapIndex(std::unique_ptr<SymbolIndex> Index = nullptr)
      : Index(std::move(Index)) {}
  void reset(std::unique_ptr<SymbolIndex>);

  // SymbolIndex methods delegate to the current index, which is kept alive
  // until the call returns (even if reset() is called).
  bool fuzzyFind(const FuzzyFindRequest &,
                 llvm::function_ref<void(const Symbol &)>) const override;
  void lookup(const LookupRequest &,
              llvm::function_ref<void(const Symbol &)>) const override;
  void refs(const RefsRequest &,
            llvm::function_ref<void(const Ref &)>) const override;
  size_t estimateMemoryUsage() const override;

private:
  std::shared_ptr<SymbolIndex> snapshot() const;
  mutable std::mutex Mutex;
  std::shared_ptr<SymbolIndex> Index;
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_INDEX_H
