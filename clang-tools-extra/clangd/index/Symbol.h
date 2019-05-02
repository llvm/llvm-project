//===--- Symbol.h ------------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_SYMBOL_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_SYMBOL_H

#include "SymbolID.h"
#include "SymbolLocation.h"
#include "SymbolOrigin.h"
#include "clang/Index/IndexSymbol.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/StringSaver.h"

namespace clang {
namespace clangd {

/// The class presents a C++ symbol, e.g. class, function.
///
/// WARNING: Symbols do not own much of their underlying data - typically
/// strings are owned by a SymbolSlab. They should be treated as non-owning
/// references. Copies are shallow.
///
/// When adding new unowned data fields to Symbol, remember to update:
///   - SymbolSlab::Builder in Index.cpp, to copy them to the slab's storage.
///   - mergeSymbol in Merge.cpp, to properly combine two Symbols.
///
/// A fully documented symbol can be split as:
/// size_type std::map<k, t>::count(const K& key) const
/// | Return  |     Scope     |Name|    Signature     |
/// We split up these components to allow display flexibility later.
struct Symbol {
  /// The ID of the symbol.
  SymbolID ID;
  /// The symbol information, like symbol kind.
  index::SymbolInfo SymInfo = index::SymbolInfo();
  /// The unqualified name of the symbol, e.g. "bar" (for ns::bar).
  llvm::StringRef Name;
  /// The containing namespace. e.g. "" (global), "ns::" (top-level namespace).
  llvm::StringRef Scope;
  /// The location of the symbol's definition, if one was found.
  /// This just covers the symbol name (e.g. without class/function body).
  SymbolLocation Definition;
  /// The location of the preferred declaration of the symbol.
  /// This just covers the symbol name.
  /// This may be the same as Definition.
  ///
  /// A C++ symbol may have multiple declarations, and we pick one to prefer.
  ///   * For classes, the canonical declaration should be the definition.
  ///   * For non-inline functions, the canonical declaration typically appears
  ///     in the ".h" file corresponding to the definition.
  SymbolLocation CanonicalDeclaration;
  /// The number of translation units that reference this symbol from their main
  /// file. This number is only meaningful if aggregated in an index.
  unsigned References = 0;
  /// Where this symbol came from. Usually an index provides a constant value.
  SymbolOrigin Origin = SymbolOrigin::Unknown;
  /// A brief description of the symbol that can be appended in the completion
  /// candidate list. For example, "(X x, Y y) const" is a function signature.
  /// Only set when the symbol is indexed for completion.
  llvm::StringRef Signature;
  /// Argument list in human-readable format, will be displayed to help
  /// disambiguate between different specializations of a template. Empty for
  /// non-specializations. Example: "<int, bool, 3>"
  llvm::StringRef TemplateSpecializationArgs;
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
    /// Symbol is an implementation detail.
    ImplementationDetail = 1 << 2,
    /// Symbol is visible to other files (not e.g. a static helper function).
    VisibleOutsideFile = 1 << 3,
  };

  SymbolFlag Flags = SymbolFlag::None;
  /// FIXME: also add deprecation message and fixit?
};

inline Symbol::SymbolFlag operator|(Symbol::SymbolFlag A,
                                    Symbol::SymbolFlag B) {
  return static_cast<Symbol::SymbolFlag>(static_cast<uint8_t>(A) |
                                         static_cast<uint8_t>(B));
}
inline Symbol::SymbolFlag &operator|=(Symbol::SymbolFlag &A,
                                      Symbol::SymbolFlag B) {
  return A = A | B;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Symbol &S);
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, Symbol::SymbolFlag);

/// Invokes Callback with each StringRef& contained in the Symbol.
/// Useful for deduplicating backing strings.
template <typename Callback> void visitStrings(Symbol &S, const Callback &CB) {
  CB(S.Name);
  CB(S.Scope);
  CB(S.TemplateSpecializationArgs);
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

/// Computes query-independent quality score for a Symbol.
/// This currently falls in the range [1, ln(#indexed documents)].
/// FIXME: this should probably be split into symbol -> signals
///        and signals -> score, so it can be reused for Sema completions.
float quality(const Symbol &S);

/// An immutable symbol container that stores a set of symbols.
/// The container will maintain the lifetime of the symbols.
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

  /// SymbolSlab::Builder is a mutable container that can 'freeze' to
  /// SymbolSlab. The frozen SymbolSlab will use less memory.
  class Builder {
  public:
    Builder() : UniqueStrings(Arena) {}

    /// Adds a symbol, overwriting any existing one with the same ID.
    /// This is a deep copy: underlying strings will be owned by the slab.
    void insert(const Symbol &S);

    /// Returns the symbol with an ID, if it exists. Valid until next insert().
    const Symbol *find(const SymbolID &ID) {
      auto I = SymbolIndex.find(ID);
      return I == SymbolIndex.end() ? nullptr : &Symbols[I->second];
    }

    /// Consumes the builder to finalize the slab.
    SymbolSlab build() &&;

  private:
    llvm::BumpPtrAllocator Arena;
    /// Intern table for strings. Contents are on the arena.
    llvm::UniqueStringSaver UniqueStrings;
    std::vector<Symbol> Symbols;
    /// Values are indices into Symbols vector.
    llvm::DenseMap<SymbolID, size_t> SymbolIndex;
  };

private:
  SymbolSlab(llvm::BumpPtrAllocator Arena, std::vector<Symbol> Symbols)
      : Arena(std::move(Arena)), Symbols(std::move(Symbols)) {}

  llvm::BumpPtrAllocator Arena; // Owns Symbol data that the Symbols do not.
  std::vector<Symbol> Symbols;  // Sorted by SymbolID to allow lookup.
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_SYMBOL_H
