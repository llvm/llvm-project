//===--- IncludeFixer.h ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INCLUDEFIXER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INCLUDEFIXER_H

#include "Diagnostics.h"
#include "Headers.h"
#include "index/Index.h"
#include "index/Symbol.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/Tooling/Inclusions/HeaderIncludes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <optional>

namespace clang {
namespace clangd {

/// Attempts to recover from error diagnostics by suggesting include insertion
/// fixes. For example, member access into incomplete type can be fixes by
/// include headers with the definition.
class IncludeFixer {
public:
  IncludeFixer(llvm::StringRef File, std::shared_ptr<IncludeInserter> Inserter,
               const SymbolIndex &Index, unsigned IndexRequestLimit,
               Symbol::IncludeDirective Directive)
      : File(File), Inserter(std::move(Inserter)), Index(Index),
        IndexRequestLimit(IndexRequestLimit), Directive(Directive) {}

  /// Returns include insertions that can potentially recover the diagnostic.
  /// If Info is a note and fixes are returned, they should *replace* the note.
  std::vector<Fix> fix(DiagnosticsEngine::Level DiagLevel,
                       const clang::Diagnostic &Info) const;

  /// Returns an ExternalSemaSource that records failed name lookups in Sema.
  /// This allows IncludeFixer to suggest inserting headers that define those
  /// names.
  llvm::IntrusiveRefCntPtr<ExternalSemaSource> unresolvedNameRecorder();

private:
  /// Attempts to recover diagnostic caused by an incomplete type \p T.
  std::vector<Fix> fixIncompleteType(const Type &T) const;

  /// Generates header insertion fixes for all symbols. Fixes are deduplicated.
  std::vector<Fix> fixesForSymbols(const SymbolSlab &Syms) const;

  std::optional<Fix> insertHeader(llvm::StringRef Name,
                                  llvm::StringRef Symbol = "",
                                  tooling::IncludeDirective Directive =
                                      tooling::IncludeDirective::Include) const;

  struct UnresolvedName {
    std::string Name;   // E.g. "X" in foo::X.
    SourceLocation Loc; // Start location of the unresolved name.
    std::vector<std::string> Scopes; // Namespace scopes we should search in.
  };

  /// Records the last unresolved name seen by Sema.
  class UnresolvedNameRecorder;

  /// Attempts to fix the unresolved name associated with the current
  /// diagnostic. We assume a diagnostic is caused by a unresolved name when
  /// they have the same source location and the unresolved name is the last
  /// one we've seen during the Sema run.
  std::vector<Fix> fixUnresolvedName() const;

  std::string File;
  std::shared_ptr<IncludeInserter> Inserter;
  const SymbolIndex &Index;
  const unsigned IndexRequestLimit; // Make at most 5 index requests.
  mutable unsigned IndexRequestCount = 0;
  const Symbol::IncludeDirective Directive;

  // These collect the last unresolved name so that we can associate it with the
  // diagnostic.
  std::optional<UnresolvedName> LastUnresolvedName;

  // There can be multiple diagnostics that are caused by the same unresolved
  // name or incomplete type in one parse, especially when code is
  // copy-and-pasted without #includes. We cache the index results based on
  // index requests.
  mutable llvm::StringMap<SymbolSlab> FuzzyFindCache;
  mutable llvm::DenseMap<SymbolID, SymbolSlab> LookupCache;
  // Returns std::nullopt if the number of index requests has reached the limit.
  std::optional<const SymbolSlab *>
  fuzzyFindCached(const FuzzyFindRequest &Req) const;
  std::optional<const SymbolSlab *> lookupCached(const SymbolID &ID) const;
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INCLUDEFIXER_H
