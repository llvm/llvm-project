//===--- MemIndex.cpp - Dynamic in-memory symbol index. ----------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-------------------------------------------------------------------===//

#include "MemIndex.h"
#include "../FuzzyMatch.h"
#include "../Logger.h"
#include <queue>

namespace clang {
namespace clangd {

std::unique_ptr<SymbolIndex> MemIndex::build(SymbolSlab Slab,
                                             SymbolOccurrenceSlab Occurrences) {
  OccurrenceMap M;
  for (const auto &SymbolAndOccurrences : Occurrences) {
    auto &Entry = M[SymbolAndOccurrences.first];
    for (const auto &Occurrence : SymbolAndOccurrences.second)
      Entry.push_back(&Occurrence);
  }
  auto Data = std::make_pair(std::move(Slab), std::move(Occurrences));
  return llvm::make_unique<MemIndex>(Data.first, std::move(M), std::move(Data));
}

bool MemIndex::fuzzyFind(
    const FuzzyFindRequest &Req,
    llvm::function_ref<void(const Symbol &)> Callback) const {
  assert(!StringRef(Req.Query).contains("::") &&
         "There must be no :: in query.");

  std::priority_queue<std::pair<float, const Symbol *>> Top;
  FuzzyMatcher Filter(Req.Query);
  bool More = false;
  for (const auto Pair : Index) {
    const Symbol *Sym = Pair.second;

    // Exact match against all possible scopes.
    if (!Req.Scopes.empty() && !llvm::is_contained(Req.Scopes, Sym->Scope))
      continue;
    if (Req.RestrictForCodeCompletion && !Sym->IsIndexedForCodeCompletion)
      continue;

    if (auto Score = Filter.match(Sym->Name)) {
      Top.emplace(-*Score * quality(*Sym), Sym);
      if (Top.size() > Req.MaxCandidateCount) {
        More = true;
        Top.pop();
      }
    }
  }
  for (; !Top.empty(); Top.pop())
    Callback(*Top.top().second);
  return More;
}

void MemIndex::lookup(const LookupRequest &Req,
                      llvm::function_ref<void(const Symbol &)> Callback) const {
  for (const auto &ID : Req.IDs) {
    auto I = Index.find(ID);
    if (I != Index.end())
      Callback(*I->second);
  }
}

void MemIndex::findOccurrences(
    const OccurrencesRequest &Req,
    llvm::function_ref<void(const SymbolOccurrence &)> Callback) const {
  for (const auto &ReqID : Req.IDs) {
    auto FoundOccurrences = Occurrences.find(ReqID);
    if (FoundOccurrences == Occurrences.end())
      continue;
    for (const auto *O : FoundOccurrences->second) {
      if (static_cast<int>(Req.Filter & O->Kind))
        Callback(*O);
    }
  }
}

size_t MemIndex::estimateMemoryUsage() const {
  return Index.getMemorySize();
}

} // namespace clangd
} // namespace clang
