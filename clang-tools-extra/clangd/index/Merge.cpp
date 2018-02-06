//===--- Merge.h ------------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
#include "Merge.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
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
     Symbol::Details Scratch;
     More |= Static->fuzzyFind(Req, [&](const Symbol &S) {
       auto DynS = Dyn.find(S.ID);
       if (DynS == Dyn.end())
         return Callback(S);
       SeenDynamicSymbols.insert(S.ID);
       Callback(mergeSymbol(*DynS, S, &Scratch));
     });
     for (const Symbol &S : Dyn)
       if (!SeenDynamicSymbols.count(S.ID))
         Callback(S);
     return !More; // returning true indicates the result is complete.
  }

private:
  const SymbolIndex *Dynamic, *Static;
};
}

Symbol
mergeSymbol(const Symbol &L, const Symbol &R, Symbol::Details *Scratch) {
  assert(L.ID == R.ID);
  Symbol S = L;
  // For each optional field, fill it from R if missing in L.
  // (It might be missing in R too, but that's a no-op).
  if (S.CanonicalDeclaration.FileURI == "")
    S.CanonicalDeclaration = R.CanonicalDeclaration;
  if (S.CompletionLabel == "")
    S.CompletionLabel = R.CompletionLabel;
  if (S.CompletionFilterText == "")
    S.CompletionFilterText = R.CompletionFilterText;
  if (S.CompletionPlainInsertText == "")
    S.CompletionPlainInsertText = R.CompletionPlainInsertText;
  if (S.CompletionSnippetInsertText == "")
    S.CompletionSnippetInsertText = R.CompletionSnippetInsertText;

  if (L.Detail && R.Detail) {
    // Copy into scratch space so we can merge.
    *Scratch = *L.Detail;
    if (Scratch->Documentation == "")
      Scratch->Documentation = R.Detail->Documentation;
    if (Scratch->CompletionDetail == "")
      Scratch->CompletionDetail = R.Detail->CompletionDetail;
    S.Detail = Scratch;
  } else if (L.Detail)
    S.Detail = L.Detail;
  else if (R.Detail)
    S.Detail = R.Detail;
  return S;
}

std::unique_ptr<SymbolIndex> mergeIndex(const SymbolIndex *Dynamic,
                                        const SymbolIndex *Static) {
  return llvm::make_unique<MergedIndex>(Dynamic, Static);
}
} // namespace clangd
} // namespace clang
