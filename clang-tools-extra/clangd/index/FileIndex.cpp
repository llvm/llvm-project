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
namespace {

/// Retrieves namespace and class level symbols in \p Decls.
std::unique_ptr<SymbolSlab> indexAST(ASTContext &Ctx,
                                     std::shared_ptr<Preprocessor> PP,
                                     llvm::ArrayRef<const Decl *> Decls) {
  SymbolCollector::Options CollectorOpts;
  // Although we do not index symbols in main files (e.g. cpp file), information
  // in main files like definition locations of class declarations will still be
  // collected; thus, the index works for go-to-definition.
  // FIXME(ioeric): get rid of `IndexMainFiles` as this is always set to false.
  CollectorOpts.IndexMainFiles = false;
  // FIXME(ioeric): we might also want to collect include headers. We would need
  // to make sure all includes are canonicalized (with CanonicalIncludes), which
  // is not trivial given the current way of collecting symbols: we only have
  // AST at this point, but we also need preprocessor callbacks (e.g.
  // CommentHandler for IWYU pragma) to canonicalize includes.
  CollectorOpts.CollectIncludePath = false;

  auto Collector = std::make_shared<SymbolCollector>(std::move(CollectorOpts));
  Collector->setPreprocessor(std::move(PP));
  index::IndexingOptions IndexOpts;
  IndexOpts.SystemSymbolFilter =
      index::IndexingOptions::SystemSymbolFilterKind::All;
  IndexOpts.IndexFunctionLocals = false;

  index::indexTopLevelDecls(Ctx, Decls, Collector, IndexOpts);
  auto Symbols = llvm::make_unique<SymbolSlab>();
  *Symbols = Collector->takeSymbols();
  return Symbols;
}

} // namespace

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
    auto Slab = indexAST(AST->getASTContext(), AST->getPreprocessorPtr(),
                         AST->getTopLevelDecls());
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

} // namespace clangd
} // namespace clang
