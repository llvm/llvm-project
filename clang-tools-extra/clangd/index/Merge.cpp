//===--- Merge.cpp -----------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Merge.h"
#include "../Logger.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

namespace clang {
namespace clangd {
namespace {

using namespace llvm;

class MergedIndex : public SymbolIndex {
 public:
   MergedIndex(const SymbolIndex *Dynamic, const SymbolIndex *Static)
       : Dynamic(Dynamic), Static(Static) {}

   // FIXME: Deleted symbols in dirty files are still returned (from Static).
   //        To identify these eliminate these, we should:
   //          - find the generating file from each Symbol which is Static-only
   //          - ask Dynamic if it has that file (needs new SymbolIndex method)
   //          - if so, drop the Symbol.
   bool fuzzyFind(const FuzzyFindRequest &Req,
                  function_ref<void(const Symbol &)> Callback) const override {
     // We can't step through both sources in parallel. So:
     //  1) query all dynamic symbols, slurping results into a slab
     //  2) query the static symbols, for each one:
     //    a) if it's not in the dynamic slab, yield it directly
     //    b) if it's in the dynamic slab, merge it and yield the result
     //  3) now yield all the dynamic symbols we haven't processed.
     bool More = false; // We'll be incomplete if either source was.
     SymbolSlab::Builder DynB;
     More |= Dynamic->fuzzyFind(Req, [&](const Symbol &S) { DynB.insert(S); });
     SymbolSlab Dyn = std::move(DynB).build();

     DenseSet<SymbolID> SeenDynamicSymbols;
     More |= Static->fuzzyFind(Req, [&](const Symbol &S) {
       auto DynS = Dyn.find(S.ID);
       if (DynS == Dyn.end())
         return Callback(S);
       SeenDynamicSymbols.insert(S.ID);
       Callback(mergeSymbol(*DynS, S));
     });
     for (const Symbol &S : Dyn)
       if (!SeenDynamicSymbols.count(S.ID))
         Callback(S);
     return More;
  }

  void
  lookup(const LookupRequest &Req,
         llvm::function_ref<void(const Symbol &)> Callback) const override {
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

  void refs(const RefsRequest &Req,
            llvm::function_ref<void(const Ref &)> Callback) const override {
    // We don't want duplicated refs from the static/dynamic indexes,
    // and we can't reliably duplicate them because offsets may differ slightly.
    // We consider the dynamic index authoritative and report all its refs,
    // and only report static index refs from other files.
    //
    // FIXME: The heuristic fails if the dynamic index contains a file, but all
    // refs were removed (we will report stale ones from the static index).
    // Ultimately we should explicit check which index has the file instead.
    llvm::StringSet<> DynamicIndexFileURIs;
    Dynamic->refs(Req, [&](const Ref &O) {
      DynamicIndexFileURIs.insert(O.Location.FileURI);
      Callback(O);
    });
    Static->refs(Req, [&](const Ref &O) {
      if (!DynamicIndexFileURIs.count(O.Location.FileURI))
        Callback(O);
    });
  }

  size_t estimateMemoryUsage() const override {
    return Dynamic->estimateMemoryUsage() + Static->estimateMemoryUsage();
  }

private:
  const SymbolIndex *Dynamic, *Static;
};
} // namespace

Symbol mergeSymbol(const Symbol &L, const Symbol &R) {
  assert(L.ID == R.ID);
  // We prefer information from TUs that saw the definition.
  // Classes: this is the def itself. Functions: hopefully the header decl.
  // If both did (or both didn't), continue to prefer L over R.
  bool PreferR = R.Definition && !L.Definition;
  // Merge include headers only if both have definitions or both have no
  // definition; otherwise, only accumulate references of common includes.
  bool MergeIncludes =
      L.Definition.FileURI.empty() == R.Definition.FileURI.empty();
  Symbol S = PreferR ? R : L;        // The target symbol we're merging into.
  const Symbol &O = PreferR ? L : R; // The "other" less-preferred symbol.

  // For each optional field, fill it from O if missing in S.
  // (It might be missing in O too, but that's a no-op).
  if (!S.Definition)
    S.Definition = O.Definition;
  if (!S.CanonicalDeclaration)
    S.CanonicalDeclaration = O.CanonicalDeclaration;
  S.References += O.References;
  if (S.Signature == "")
    S.Signature = O.Signature;
  if (S.CompletionSnippetSuffix == "")
    S.CompletionSnippetSuffix = O.CompletionSnippetSuffix;
  if (S.Documentation == "")
    S.Documentation = O.Documentation;
  if (S.ReturnType == "")
    S.ReturnType = O.ReturnType;
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
  return S;
}

std::unique_ptr<SymbolIndex> mergeIndex(const SymbolIndex *Dynamic,
                                        const SymbolIndex *Static) {
  return llvm::make_unique<MergedIndex>(Dynamic, Static);
}

} // namespace clangd
} // namespace clang
