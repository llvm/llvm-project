//===--- Index.cpp -----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Index.h"
#include "support/Logger.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>

namespace clang {
namespace clangd {

void SwapIndex::reset(std::unique_ptr<SymbolIndex> Index) {
  // Keep the old index alive, so we don't destroy it under lock (may be slow).
  std::shared_ptr<SymbolIndex> Pin;
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    Pin = std::move(this->Index);
    this->Index = std::move(Index);
  }
}
std::shared_ptr<SymbolIndex> SwapIndex::snapshot() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  return Index;
}

bool fromJSON(const llvm::json::Value &Parameters, FuzzyFindRequest &Request,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Parameters, P);
  int64_t Limit;
  bool OK =
      O && O.map("Query", Request.Query) && O.map("Scopes", Request.Scopes) &&
      O.map("AnyScope", Request.AnyScope) && O.map("Limit", Limit) &&
      O.map("RestrictForCodeCompletion", Request.RestrictForCodeCompletion) &&
      O.map("ProximityPaths", Request.ProximityPaths) &&
      O.map("PreferredTypes", Request.PreferredTypes);
  if (OK && Limit <= std::numeric_limits<uint32_t>::max())
    Request.Limit = Limit;
  return OK;
}

llvm::json::Value toJSON(const FuzzyFindRequest &Request) {
  return llvm::json::Object{
      {"Query", Request.Query},
      {"Scopes", Request.Scopes},
      {"AnyScope", Request.AnyScope},
      {"Limit", Request.Limit},
      {"RestrictForCodeCompletion", Request.RestrictForCodeCompletion},
      {"ProximityPaths", Request.ProximityPaths},
      {"PreferredTypes", Request.PreferredTypes},
  };
}

bool SwapIndex::fuzzyFind(const FuzzyFindRequest &R,
                          llvm::function_ref<void(const Symbol &)> CB) const {
  return snapshot()->fuzzyFind(R, CB);
}
void SwapIndex::lookup(const LookupRequest &R,
                       llvm::function_ref<void(const Symbol &)> CB) const {
  return snapshot()->lookup(R, CB);
}
bool SwapIndex::refs(const RefsRequest &R,
                     llvm::function_ref<void(const Ref &)> CB) const {
  return snapshot()->refs(R, CB);
}
void SwapIndex::relations(
    const RelationsRequest &R,
    llvm::function_ref<void(const SymbolID &, const Symbol &)> CB) const {
  return snapshot()->relations(R, CB);
}

llvm::unique_function<IndexContents(llvm::StringRef) const>
SwapIndex::indexedFiles() const {
  // The index snapshot should outlive this method return value.
  auto SnapShot = snapshot();
  auto IndexedFiles = SnapShot->indexedFiles();
  return [KeepAlive{std::move(SnapShot)},
          IndexContainsFile{std::move(IndexedFiles)}](llvm::StringRef File) {
    return IndexContainsFile(File);
  };
}

size_t SwapIndex::estimateMemoryUsage() const {
  return snapshot()->estimateMemoryUsage();
}

} // namespace clangd
} // namespace clang
