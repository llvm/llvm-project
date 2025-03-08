//===--- Index.h -------------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_INDEX_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_INDEX_H

#include "index/Ref.h"
#include "index/Relation.h"
#include "index/Symbol.h"
#include "index/SymbolID.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/JSON.h"
#include <mutex>
#include <optional>
#include <string>

namespace clang {
namespace clangd {

struct FuzzyFindRequest {
  /// A query string for the fuzzy find. This is matched against symbols'
  /// un-qualified identifiers and should not contain qualifiers like "::".
  std::string Query;
  /// If this is non-empty, symbols must be in at least one of the scopes
  /// (e.g. namespaces) excluding nested scopes. For example, if a scope "xyz::"
  /// is provided, the matched symbols must be defined in namespace xyz but not
  /// namespace xyz::abc.
  ///
  /// The global scope is "", a top level scope is "foo::", etc.
  std::vector<std::string> Scopes;
  /// If set to true, allow symbols from any scope. Scopes explicitly listed
  /// above will be ranked higher.
  bool AnyScope = false;
  /// The number of top candidates to return. The index may choose to
  /// return more than this, e.g. if it doesn't know which candidates are best.
  std::optional<uint32_t> Limit;
  /// If set to true, only symbols for completion support will be considered.
  bool RestrictForCodeCompletion = false;
  /// Contextually relevant files (e.g. the file we're code-completing in).
  /// Paths should be absolute.
  std::vector<std::string> ProximityPaths;
  /// Preferred types of symbols. These are raw representation of `OpaqueType`.
  std::vector<std::string> PreferredTypes;

  bool operator==(const FuzzyFindRequest &Req) const {
    return std::tie(Query, Scopes, Limit, RestrictForCodeCompletion,
                    ProximityPaths, PreferredTypes) ==
           std::tie(Req.Query, Req.Scopes, Req.Limit,
                    Req.RestrictForCodeCompletion, Req.ProximityPaths,
                    Req.PreferredTypes);
  }
  bool operator!=(const FuzzyFindRequest &Req) const { return !(*this == Req); }
};
bool fromJSON(const llvm::json::Value &Value, FuzzyFindRequest &Request,
              llvm::json::Path);
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
  std::optional<uint32_t> Limit;
  /// If set, populates the container of the reference.
  /// Index implementations may chose to populate containers no matter what.
  bool WantContainer = false;
};

struct ContainedRefsRequest {
  /// Note that RefKind::Call just restricts the matched SymbolKind to
  /// functions, not the form of the reference (e.g. address-of-function,
  /// which can indicate an indirect call, should still be caught).
  static const RefKind SupportedRefKinds = RefKind::Call;

  SymbolID ID;
  /// If set, limit the number of refers returned from the index. The index may
  /// choose to return less than this, e.g. it tries to avoid returning stale
  /// results.
  std::optional<uint32_t> Limit;
};

struct RelationsRequest {
  llvm::DenseSet<SymbolID> Subjects;
  RelationKind Predicate;
  /// If set, limit the number of relations returned from the index.
  std::optional<uint32_t> Limit;
};

struct ContainedRefsResult {
  /// The source location where the symbol is named.
  SymbolLocation Location;
  RefKind Kind = RefKind::Unknown;
  /// The ID of the symbol which is referred to
  SymbolID Symbol;
};

/// Describes what data is covered by an index.
///
/// Indexes may contain symbols but not references from a file, etc.
/// This affects merging: if a staler index contains a reference but a fresher
/// one does not, we want to trust the fresher index *only* if it actually
/// includes references in general.
enum class IndexContents : uint8_t {
  None = 0,
  Symbols = 1 << 1,
  References = 1 << 2,
  Relations = 1 << 3,
  All = Symbols | References | Relations
};

inline constexpr IndexContents operator&(IndexContents L, IndexContents R) {
  return static_cast<IndexContents>(static_cast<uint8_t>(L) &
                                    static_cast<uint8_t>(R));
}

inline constexpr IndexContents operator|(IndexContents L, IndexContents R) {
  return static_cast<IndexContents>(static_cast<uint8_t>(L) |
                                    static_cast<uint8_t>(R));
}

/// Interface for symbol indexes that can be used for searching or
/// matching symbols among a set of symbols based on names or unique IDs.
class SymbolIndex {
public:
  virtual ~SymbolIndex() = default;

  /// Matches symbols in the index fuzzily and applies \p Callback on
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

  /// Finds all occurrences (e.g. references, declarations, definitions) of
  /// symbols and applies \p Callback on each result.
  ///
  /// Results should be returned in arbitrary order.
  /// The returned result must be deep-copied if it's used outside Callback.
  /// FIXME: there's no indication which result references which symbol.
  ///
  /// Returns true if there will be more results (limited by Req.Limit);
  virtual bool refs(const RefsRequest &Req,
                    llvm::function_ref<void(const Ref &)> Callback) const = 0;

  /// Find all symbols that are referenced by a symbol and apply
  /// \p Callback on each result.
  ///
  /// Results should be returned in arbitrary order.
  /// The returned result must be deep-copied if it's used outside Callback.
  ///
  /// Returns true if there will be more results (limited by Req.Limit);
  virtual bool containedRefs(
      const ContainedRefsRequest &Req,
      llvm::function_ref<void(const ContainedRefsResult &)> Callback) const = 0;

  /// Finds all relations (S, P, O) stored in the index such that S is among
  /// Req.Subjects and P is Req.Predicate, and invokes \p Callback for (S, O) in
  /// each.
  virtual void relations(
      const RelationsRequest &Req,
      llvm::function_ref<void(const SymbolID &Subject, const Symbol &Object)>
          Callback) const = 0;

  /// Returns function which checks if the specified file was used to build this
  /// index or not. The function must only be called while the index is alive.
  using IndexedFiles =
      llvm::unique_function<IndexContents(llvm::StringRef) const>;
  virtual IndexedFiles indexedFiles() const = 0;

  /// Returns estimated size of index (in bytes).
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
  bool refs(const RefsRequest &,
            llvm::function_ref<void(const Ref &)>) const override;
  bool containedRefs(
      const ContainedRefsRequest &,
      llvm::function_ref<void(const ContainedRefsResult &)>) const override;
  void relations(const RelationsRequest &,
                 llvm::function_ref<void(const SymbolID &, const Symbol &)>)
      const override;

  llvm::unique_function<IndexContents(llvm::StringRef) const>
  indexedFiles() const override;

  size_t estimateMemoryUsage() const override;

private:
  std::shared_ptr<SymbolIndex> snapshot() const;
  mutable std::mutex Mutex;
  std::shared_ptr<SymbolIndex> Index;
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_INDEX_H
