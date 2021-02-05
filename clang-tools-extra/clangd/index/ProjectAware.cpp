//===--- ProjectAware.h ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProjectAware.h"
#include "Config.h"
#include "index/Index.h"
#include "index/MemIndex.h"
#include "index/Merge.h"
#include "index/Ref.h"
#include "index/Serialization.h"
#include "index/Symbol.h"
#include "index/SymbolID.h"
#include "support/Logger.h"
#include "support/Threading.h"
#include "support/Trace.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

namespace clang {
namespace clangd {
namespace {
class ProjectAwareIndex : public SymbolIndex {
public:
  size_t estimateMemoryUsage() const override;

  /// Only queries the associated index with the current context.
  void lookup(const LookupRequest &Req,
              llvm::function_ref<void(const Symbol &)> Callback) const override;

  /// Query all indexes while prioritizing the associated one (if any).
  bool refs(const RefsRequest &Req,
            llvm::function_ref<void(const Ref &)> Callback) const override;

  /// Queries only the associates index when Req.RestrictForCodeCompletion is
  /// set, otherwise queries all.
  bool
  fuzzyFind(const FuzzyFindRequest &Req,
            llvm::function_ref<void(const Symbol &)> Callback) const override;

  /// Query all indexes while prioritizing the associated one (if any).
  void relations(const RelationsRequest &Req,
                 llvm::function_ref<void(const SymbolID &, const Symbol &)>
                     Callback) const override;

  llvm::unique_function<IndexContents(llvm::StringRef) const>
  indexedFiles() const override;

  ProjectAwareIndex(IndexFactory Gen) : Gen(std::move(Gen)) {}

private:
  // Returns the index associated with current context, if any.
  SymbolIndex *getIndex() const;

  // Storage for all the external indexes.
  mutable std::mutex Mu;
  mutable llvm::DenseMap<Config::ExternalIndexSpec,
                         std::unique_ptr<SymbolIndex>>
      IndexForSpec;
  mutable AsyncTaskRunner Tasks;

  const IndexFactory Gen;
};

size_t ProjectAwareIndex::estimateMemoryUsage() const {
  size_t Total = 0;
  std::lock_guard<std::mutex> Lock(Mu);
  for (auto &Entry : IndexForSpec)
    Total += Entry.second->estimateMemoryUsage();
  return Total;
}

void ProjectAwareIndex::lookup(
    const LookupRequest &Req,
    llvm::function_ref<void(const Symbol &)> Callback) const {
  trace::Span Tracer("ProjectAwareIndex::lookup");
  if (auto *Idx = getIndex())
    Idx->lookup(Req, Callback);
}

bool ProjectAwareIndex::refs(
    const RefsRequest &Req,
    llvm::function_ref<void(const Ref &)> Callback) const {
  trace::Span Tracer("ProjectAwareIndex::refs");
  if (auto *Idx = getIndex())
    return Idx->refs(Req, Callback);
  return false;
}

bool ProjectAwareIndex::fuzzyFind(
    const FuzzyFindRequest &Req,
    llvm::function_ref<void(const Symbol &)> Callback) const {
  trace::Span Tracer("ProjectAwareIndex::fuzzyFind");
  if (auto *Idx = getIndex())
    return Idx->fuzzyFind(Req, Callback);
  return false;
}

void ProjectAwareIndex::relations(
    const RelationsRequest &Req,
    llvm::function_ref<void(const SymbolID &, const Symbol &)> Callback) const {
  trace::Span Tracer("ProjectAwareIndex::relations");
  if (auto *Idx = getIndex())
    return Idx->relations(Req, Callback);
}

llvm::unique_function<IndexContents(llvm::StringRef) const>
ProjectAwareIndex::indexedFiles() const {
  trace::Span Tracer("ProjectAwareIndex::indexedFiles");
  if (auto *Idx = getIndex())
    return Idx->indexedFiles();
  return [](llvm::StringRef) { return IndexContents::None; };
}

SymbolIndex *ProjectAwareIndex::getIndex() const {
  const auto &C = Config::current();
  if (!C.Index.External)
    return nullptr;
  const auto &External = *C.Index.External;
  std::lock_guard<std::mutex> Lock(Mu);
  auto Entry = IndexForSpec.try_emplace(External, nullptr);
  if (Entry.second)
    Entry.first->getSecond() = Gen(External, Tasks);
  return Entry.first->second.get();
}
} // namespace

std::unique_ptr<SymbolIndex> createProjectAwareIndex(IndexFactory Gen) {
  assert(Gen);
  return std::make_unique<ProjectAwareIndex>(std::move(Gen));
}
} // namespace clangd
} // namespace clang
