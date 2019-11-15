//===--- Merge.cpp -----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Merge.h"
#include "Logger.h"
#include "Trace.h"
#include "index/Symbol.h"
#include "index/SymbolLocation.h"
#include "index/SymbolOrigin.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <iterator>

namespace clang {
namespace clangd {

// FIXME: Deleted symbols in dirty files are still returned (from Static).
//        To identify these eliminate these, we should:
//          - find the generating file from each Symbol which is Static-only
//          - ask Dynamic if it has that file (needs new SymbolIndex method)
//          - if so, drop the Symbol.
bool MergedIndex::fuzzyFind(
    const FuzzyFindRequest &Req,
    llvm::function_ref<void(const Symbol &)> Callback) const {
  // We can't step through both sources in parallel. So:
  //  1) query all dynamic symbols, slurping results into a slab
  //  2) query the static symbols, for each one:
  //    a) if it's not in the dynamic slab, yield it directly
  //    b) if it's in the dynamic slab, merge it and yield the result
  //  3) now yield all the dynamic symbols we haven't processed.
  trace::Span Tracer("MergedIndex fuzzyFind");
  bool More = false; // We'll be incomplete if either source was.
  SymbolSlab::Builder DynB;
  unsigned DynamicCount = 0;
  unsigned StaticCount = 0;
  unsigned MergedCount = 0;
  More |= Dynamic->fuzzyFind(Req, [&](const Symbol &S) {
    ++DynamicCount;
    DynB.insert(S);
  });
  SymbolSlab Dyn = std::move(DynB).build();

  llvm::DenseSet<SymbolID> SeenDynamicSymbols;
  More |= Static->fuzzyFind(Req, [&](const Symbol &S) {
    auto DynS = Dyn.find(S.ID);
    ++StaticCount;
    if (DynS == Dyn.end())
      return Callback(S);
    ++MergedCount;
    SeenDynamicSymbols.insert(S.ID);
    Callback(mergeSymbol(*DynS, S));
  });
  SPAN_ATTACH(Tracer, "dynamic", DynamicCount);
  SPAN_ATTACH(Tracer, "static", StaticCount);
  SPAN_ATTACH(Tracer, "merged", MergedCount);
  for (const Symbol &S : Dyn)
    if (!SeenDynamicSymbols.count(S.ID))
      Callback(S);
  return More;
}

void MergedIndex::lookup(
    const LookupRequest &Req,
    llvm::function_ref<void(const Symbol &)> Callback) const {
  trace::Span Tracer("MergedIndex lookup");
  SymbolSlab::Builder B;

  Dynamic->lookup(Req, [&](const Symbol &S) { B.insert(S); });

  auto RemainingIDs = Req.IDs;
  Static->lookup(Req, [&](const Symbol &S) {
    const Symbol *Sym = B.find(S.ID);
    RemainingIDs.erase(S.ID);
    if (!Sym)
      Callback(S);
    else
      Callback(mergeSymbol(*Sym, S));
  });
  for (const auto &ID : RemainingIDs)
    if (const Symbol *Sym = B.find(ID))
      Callback(*Sym);
}

bool MergedIndex::refs(const RefsRequest &Req,
                       llvm::function_ref<void(const Ref &)> Callback) const {
  trace::Span Tracer("MergedIndex refs");
  bool More = false;
  uint32_t Remaining =
      Req.Limit.getValueOr(std::numeric_limits<uint32_t>::max());
  // We don't want duplicated refs from the static/dynamic indexes,
  // and we can't reliably deduplicate them because offsets may differ slightly.
  // We consider the dynamic index authoritative and report all its refs,
  // and only report static index refs from other files.
  //
  // FIXME: The heuristic fails if the dynamic index contains a file, but all
  // refs were removed (we will report stale ones from the static index).
  // Ultimately we should explicit check which index has the file instead.
  llvm::StringSet<> DynamicIndexFileURIs;
  More |= Dynamic->refs(Req, [&](const Ref &O) {
    DynamicIndexFileURIs.insert(O.Location.FileURI);
    Callback(O);
    assert(Remaining != 0);
    --Remaining;
  });
  if (Remaining == 0 && More)
    return More;
  // We return less than Req.Limit if static index returns more refs for dirty
  // files.
  bool StaticHadMore =  Static->refs(Req, [&](const Ref &O) {
    if (DynamicIndexFileURIs.count(O.Location.FileURI))
      return; // ignore refs that have been seen from dynamic index.
    if (Remaining == 0) {
      More = true;
      return;
    }
    --Remaining;
    Callback(O);
  });
  return More || StaticHadMore;
}

void MergedIndex::relations(
    const RelationsRequest &Req,
    llvm::function_ref<void(const SymbolID &, const Symbol &)> Callback) const {
  uint32_t Remaining =
      Req.Limit.getValueOr(std::numeric_limits<uint32_t>::max());
  // Return results from both indexes but avoid duplicates.
  // We might return stale relations from the static index;
  // we don't currently have a good way of identifying them.
  llvm::DenseSet<std::pair<SymbolID, SymbolID>> SeenRelations;
  Dynamic->relations(Req, [&](const SymbolID &Subject, const Symbol &Object) {
    Callback(Subject, Object);
    SeenRelations.insert(std::make_pair(Subject, Object.ID));
    --Remaining;
  });
  if (Remaining == 0)
    return;
  Static->relations(Req, [&](const SymbolID &Subject, const Symbol &Object) {
    if (Remaining > 0 &&
        !SeenRelations.count(std::make_pair(Subject, Object.ID))) {
      --Remaining;
      Callback(Subject, Object);
    }
  });
}

// Returns true if \p L is (strictly) preferred to \p R (e.g. by file paths). If
// neither is preferred, this returns false.
bool prefer(const SymbolLocation &L, const SymbolLocation &R) {
  if (!L)
    return false;
  if (!R)
    return true;
  auto HasCodeGenSuffix = [](const SymbolLocation &Loc) {
    constexpr static const char *CodegenSuffixes[] = {".proto"};
    return std::any_of(std::begin(CodegenSuffixes), std::end(CodegenSuffixes),
                       [&](llvm::StringRef Suffix) {
                         return llvm::StringRef(Loc.FileURI).endswith(Suffix);
                       });
  };
  return HasCodeGenSuffix(L) && !HasCodeGenSuffix(R);
}

Symbol mergeSymbol(const Symbol &L, const Symbol &R) {
  assert(L.ID == R.ID);
  // We prefer information from TUs that saw the definition.
  // Classes: this is the def itself. Functions: hopefully the header decl.
  // If both did (or both didn't), continue to prefer L over R.
  bool PreferR = R.Definition && !L.Definition;
  // Merge include headers only if both have definitions or both have no
  // definition; otherwise, only accumulate references of common includes.
  assert(L.Definition.FileURI && R.Definition.FileURI);
  bool MergeIncludes =
      bool(*L.Definition.FileURI) == bool(*R.Definition.FileURI);
  Symbol S = PreferR ? R : L;        // The target symbol we're merging into.
  const Symbol &O = PreferR ? L : R; // The "other" less-preferred symbol.

  // Only use locations in \p O if it's (strictly) preferred.
  if (prefer(O.CanonicalDeclaration, S.CanonicalDeclaration))
    S.CanonicalDeclaration = O.CanonicalDeclaration;
  if (prefer(O.Definition, S.Definition))
    S.Definition = O.Definition;
  S.References += O.References;
  if (S.Signature == "")
    S.Signature = O.Signature;
  if (S.CompletionSnippetSuffix == "")
    S.CompletionSnippetSuffix = O.CompletionSnippetSuffix;
  if (S.Documentation == "") {
    // Don't accept documentation from bare forward class declarations, if there
    // is a definition and it didn't provide one. S is often an undocumented
    // class, and O is a non-canonical forward decl preceded by an irrelevant
    // comment.
    bool IsClass = S.SymInfo.Kind == index::SymbolKind::Class ||
                   S.SymInfo.Kind == index::SymbolKind::Struct ||
                   S.SymInfo.Kind == index::SymbolKind::Union;
    if (!IsClass || !S.Definition)
      S.Documentation = O.Documentation;
  }
  if (S.ReturnType == "")
    S.ReturnType = O.ReturnType;
  if (S.Type == "")
    S.Type = O.Type;
  for (const auto &OI : O.IncludeHeaders) {
    bool Found = false;
    for (auto &SI : S.IncludeHeaders) {
      if (SI.IncludeHeader == OI.IncludeHeader) {
        Found = true;
        SI.References += OI.References;
        break;
      }
    }
    if (!Found && MergeIncludes)
      S.IncludeHeaders.emplace_back(OI.IncludeHeader, OI.References);
  }

  S.Origin |= O.Origin | SymbolOrigin::Merge;
  S.Flags |= O.Flags;
  return S;
}

} // namespace clangd
} // namespace clang
