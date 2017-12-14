//===--- MemIndex.cpp - Dynamic in-memory symbol index. ----------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-------------------------------------------------------------------===//

#include "MemIndex.h"

namespace clang {
namespace clangd {

void MemIndex::build(std::shared_ptr<std::vector<const Symbol *>> Syms) {
  llvm::DenseMap<SymbolID, const Symbol *> TempIndex;
  for (const Symbol *Sym : *Syms)
    TempIndex[Sym->ID] = Sym;

  // Swap out the old symbols and index.
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    Index = std::move(TempIndex);
    Symbols = std::move(Syms); // Relase old symbols.
  }
}

bool MemIndex::fuzzyFind(Context & /*Ctx*/, const FuzzyFindRequest &Req,
                         std::function<void(const Symbol &)> Callback) const {
  std::string LoweredQuery = llvm::StringRef(Req.Query).lower();
  unsigned Matched = 0;
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    for (const auto Pair : Index) {
      const Symbol *Sym = Pair.second;
      // Find all symbols that contain the query, igoring cases.
      // FIXME: consider matching chunks in qualified names instead the whole
      // string.
      // FIXME: use better matching algorithm, e.g. fuzzy matcher.
      if (StringRef(StringRef(Sym->QualifiedName).lower())
              .contains(LoweredQuery)) {
        if (++Matched > Req.MaxCandidateCount)
          return false;
        Callback(*Sym);
      }
    }
  }
  return true;
}

} // namespace clangd
} // namespace clang
