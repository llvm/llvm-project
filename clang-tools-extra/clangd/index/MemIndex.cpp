//===--- MemIndex.cpp - Dynamic in-memory symbol index. ----------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------===//

#include "MemIndex.h"
#include "FuzzyMatch.h"
#include "Logger.h"
#include "Quality.h"
#include "Trace.h"

namespace clang {
namespace clangd {

std::unique_ptr<SymbolIndex> MemIndex::build(SymbolSlab Slab, RefSlab Refs) {
  // Store Slab size before it is moved.
  const auto BackingDataSize = Slab.bytes() + Refs.bytes();
  auto Data = std::make_pair(std::move(Slab), std::move(Refs));
  return llvm::make_unique<MemIndex>(Data.first, Data.second, std::move(Data),
                                     BackingDataSize);
}

bool MemIndex::fuzzyFind(
    const FuzzyFindRequest &Req,
    llvm::function_ref<void(const Symbol &)> Callback) const {
  assert(!StringRef(Req.Query).contains("::") &&
         "There must be no :: in query.");
  trace::Span Tracer("MemIndex fuzzyFind");

  TopN<std::pair<float, const Symbol *>> Top(
      Req.Limit ? *Req.Limit : std::numeric_limits<size_t>::max());
  FuzzyMatcher Filter(Req.Query);
  bool More = false;
  for (const auto Pair : Index) {
    const Symbol *Sym = Pair.second;

    // Exact match against all possible scopes.
    if (!Req.AnyScope && !llvm::is_contained(Req.Scopes, Sym->Scope))
      continue;
    if (Req.RestrictForCodeCompletion &&
        !(Sym->Flags & Symbol::IndexedForCodeCompletion))
      continue;

    if (auto Score = Filter.match(Sym->Name))
      if (Top.push({*Score * quality(*Sym), Sym}))
        More = true; // An element with smallest score was discarded.
  }
  auto Results = std::move(Top).items();
  SPAN_ATTACH(Tracer, "results", static_cast<int>(Results.size()));
  for (const auto &Item : Results)
    Callback(*Item.second);
  return More;
}

void MemIndex::lookup(const LookupRequest &Req,
                      llvm::function_ref<void(const Symbol &)> Callback) const {
  trace::Span Tracer("MemIndex lookup");
  for (const auto &ID : Req.IDs) {
    auto I = Index.find(ID);
    if (I != Index.end())
      Callback(*I->second);
  }
}

void MemIndex::refs(const RefsRequest &Req,
                    llvm::function_ref<void(const Ref &)> Callback) const {
  trace::Span Tracer("MemIndex refs");
  uint32_t Remaining =
      Req.Limit.getValueOr(std::numeric_limits<uint32_t>::max());
  for (const auto &ReqID : Req.IDs) {
    auto SymRefs = Refs.find(ReqID);
    if (SymRefs == Refs.end())
      continue;
    for (const auto &O : SymRefs->second) {
      if (Remaining > 0 && static_cast<int>(Req.Filter & O.Kind)) {
        --Remaining;
        Callback(O);
      }
    }
  }
}

size_t MemIndex::estimateMemoryUsage() const {
  return Index.getMemorySize() + Refs.getMemorySize() + BackingDataSize;
}

} // namespace clangd
} // namespace clang
