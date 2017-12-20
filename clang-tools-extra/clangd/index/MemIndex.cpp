//===--- MemIndex.cpp - Dynamic in-memory symbol index. ----------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-------------------------------------------------------------------===//

#include "MemIndex.h"
#include "../Logger.h"

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

bool MemIndex::fuzzyFind(const Context &Ctx, const FuzzyFindRequest &Req,
                         std::function<void(const Symbol &)> Callback) const {
  assert(!StringRef(Req.Query).contains("::") &&
         "There must be no :: in query.");

  unsigned Matched = 0;
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    for (const auto Pair : Index) {
      const Symbol *Sym = Pair.second;

      // Exact match against all possible scopes.
      bool ScopeMatched = Req.Scopes.empty();
      for (StringRef Scope : Req.Scopes) {
        if (Scope == Sym->Scope) {
          ScopeMatched = true;
          break;
        }
      }
      if (!ScopeMatched)
        continue;

      // FIXME(ioeric): use fuzzy matcher.
      if (StringRef(Sym->Name).find_lower(Req.Query) != StringRef::npos) {
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
