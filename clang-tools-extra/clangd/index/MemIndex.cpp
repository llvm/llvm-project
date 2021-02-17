//===--- MemIndex.cpp - Dynamic in-memory symbol index. ----------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------===//

#include "MemIndex.h"
#include "FuzzyMatch.h"
#include "Quality.h"
#include "support/Logger.h"
#include "support/Trace.h"
#include "clang/Index/IndexSymbol.h"

namespace clang {
namespace clangd {

std::unique_ptr<SymbolIndex> MemIndex::build(SymbolSlab Slab, RefSlab Refs,
                                             RelationSlab Relations) {
  // Store Slab size before it is moved.
  const auto BackingDataSize = Slab.bytes() + Refs.bytes();
  auto Data = std::make_pair(std::move(Slab), std::move(Refs));
  return std::make_unique<MemIndex>(Data.first, Data.second, Relations,
                                     std::move(Data), BackingDataSize);
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
  for (const auto &Pair : Index) {
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

bool MemIndex::refs(const RefsRequest &Req,
                    llvm::function_ref<void(const Ref &)> Callback) const {
  trace::Span Tracer("MemIndex refs");
  uint32_t Remaining =
      Req.Limit.getValueOr(std::numeric_limits<uint32_t>::max());
  for (const auto &ReqID : Req.IDs) {
    auto SymRefs = Refs.find(ReqID);
    if (SymRefs == Refs.end())
      continue;
    for (const auto &O : SymRefs->second) {
      if (!static_cast<int>(Req.Filter & O.Kind))
        continue;
      if (Remaining == 0)
        return true; // More refs were available.
      --Remaining;
      Callback(O);
    }
  }
  return false; // We reported all refs.
}

void MemIndex::relations(
    const RelationsRequest &Req,
    llvm::function_ref<void(const SymbolID &, const Symbol &)> Callback) const {
  uint32_t Remaining =
      Req.Limit.getValueOr(std::numeric_limits<uint32_t>::max());
  for (const SymbolID &Subject : Req.Subjects) {
    LookupRequest LookupReq;
    auto It = Relations.find(
        std::make_pair(Subject, static_cast<uint8_t>(Req.Predicate)));
    if (It != Relations.end()) {
      for (const auto &Obj : It->second) {
        if (Remaining > 0) {
          --Remaining;
          LookupReq.IDs.insert(Obj);
        }
      }
    }
    lookup(LookupReq, [&](const Symbol &Object) { Callback(Subject, Object); });
  }
}

llvm::unique_function<IndexContents(llvm::StringRef) const>
MemIndex::indexedFiles() const {
  return [this](llvm::StringRef FileURI) {
    if (Files.empty())
      return IndexContents::None;
    auto Path = URI::resolve(FileURI, Files.begin()->first());
    if (!Path) {
      vlog("Failed to resolve the URI {0} : {1}", FileURI, Path.takeError());
      return IndexContents::None;
    }
    return Files.contains(*Path) ? IdxContents : IndexContents::None;
  };
}

size_t MemIndex::estimateMemoryUsage() const {
  return Index.getMemorySize() + Refs.getMemorySize() +
         Relations.getMemorySize() + BackingDataSize;
}

} // namespace clangd
} // namespace clang
