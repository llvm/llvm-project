//===--- Index.h -------------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_INDEX_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_INDEX_H

#include "Ref.h"
#include "Relation.h"
#include "Symbol.h"
#include "SymbolID.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include <mutex>
#include <string>

namespace clang {
namespace clangd {

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
