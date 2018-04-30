//===--- FileIndex.cpp - Indexes for files. ------------------------ C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FileIndex.h"
#include "SymbolCollector.h"
#include "clang/Index/IndexingAction.h"

namespace clang {
namespace clangd {

SymbolSlab indexAST(ParsedAST *AST) {
  assert(AST && "AST must not be nullptr!");
  SymbolCollector::Options CollectorOpts;
  // FIXME(ioeric): we might also want to collect include headers. We would need
  // to make sure all includes are canonicalized (with CanonicalIncludes), which
  // is not trivial given the current way of collecting symbols: we only have
  // AST at this point, but we also need preprocessor callbacks (e.g.
  // CommentHandler for IWYU pragma) to canonicalize includes.
  CollectorOpts.CollectIncludePath = false;
  CollectorOpts.CountReferences = false;

  SymbolCollector Collector(std::move(CollectorOpts));
  Collector.setPreprocessor(AST->getPreprocessorPtr());
  index::IndexingOptions IndexOpts;
  // We only need declarations, because we don't count references.
  IndexOpts.SystemSymbolFilter =
      index::IndexingOptions::SystemSymbolFilterKind::DeclarationsOnly;
  IndexOpts.IndexFunctionLocals = false;

  index::indexTopLevelDecls(AST->getASTContext(), AST->getTopLevelDecls(),
                            Collector, IndexOpts);
  return Collector.takeSymbols();
}

void FileSymbols::update(PathRef Path, std::unique_ptr<SymbolSlab> Slab) {
  std::lock_guard<std::mutex> Lock(Mutex);
  if (!Slab)
    FileToSlabs.erase(Path);
  else
    FileToSlabs[Path] = std::move(Slab);
}

std::shared_ptr<std::vector<const Symbol *>> FileSymbols::allSymbols() {
  // The snapshot manages life time of symbol slabs and provides pointers of all
  // symbols in all slabs.
  struct Snapshot {
    std::vector<const Symbol *> Pointers;
    std::vector<std::shared_ptr<SymbolSlab>> KeepAlive;
  };
  auto Snap = std::make_shared<Snapshot>();
  {
    std::lock_guard<std::mutex> Lock(Mutex);

    for (const auto &FileAndSlab : FileToSlabs) {
      Snap->KeepAlive.push_back(FileAndSlab.second);
      for (const auto &Iter : *FileAndSlab.second)
        Snap->Pointers.push_back(&Iter);
    }
  }
  auto *Pointers = &Snap->Pointers;
  // Use aliasing constructor to keep the snapshot alive along with the
  // pointers.
  return {std::move(Snap), Pointers};
}

void FileIndex::update(PathRef Path, ParsedAST *AST) {
  if (!AST) {
    FSymbols.update(Path, nullptr);
  } else {
    auto Slab = llvm::make_unique<SymbolSlab>();
    *Slab = indexAST(AST);
    FSymbols.update(Path, std::move(Slab));
  }
  auto Symbols = FSymbols.allSymbols();
  Index.build(std::move(Symbols));
}

bool FileIndex::fuzzyFind(
    const FuzzyFindRequest &Req,
    llvm::function_ref<void(const Symbol &)> Callback) const {
  return Index.fuzzyFind(Req, Callback);
}

void FileIndex::lookup(
    const LookupRequest &Req,
    llvm::function_ref<void(const Symbol &)> Callback) const {
  Index.lookup(Req, Callback);
}

} // namespace clangd
} // namespace clang
