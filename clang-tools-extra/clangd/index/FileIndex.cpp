//===--- FileIndex.cpp - Indexes for files. ------------------------ C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FileIndex.h"
#include "../Logger.h"
#include "SymbolCollector.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Lex/Preprocessor.h"

namespace clang {
namespace clangd {

std::pair<SymbolSlab, RefSlab>
indexAST(ASTContext &AST, std::shared_ptr<Preprocessor> PP,
         llvm::Optional<llvm::ArrayRef<Decl *>> TopLevelDecls,
         llvm::ArrayRef<std::string> URISchemes) {
  SymbolCollector::Options CollectorOpts;
  // FIXME(ioeric): we might also want to collect include headers. We would need
  // to make sure all includes are canonicalized (with CanonicalIncludes), which
  // is not trivial given the current way of collecting symbols: we only have
  // AST at this point, but we also need preprocessor callbacks (e.g.
  // CommentHandler for IWYU pragma) to canonicalize includes.
  CollectorOpts.CollectIncludePath = false;
  CollectorOpts.CountReferences = false;
  if (!URISchemes.empty())
    CollectorOpts.URISchemes = URISchemes;
  CollectorOpts.Origin = SymbolOrigin::Dynamic;

  index::IndexingOptions IndexOpts;
  // We only need declarations, because we don't count references.
  IndexOpts.SystemSymbolFilter =
      index::IndexingOptions::SystemSymbolFilterKind::DeclarationsOnly;
  IndexOpts.IndexFunctionLocals = false;

  std::vector<Decl *> DeclsToIndex;
  if (TopLevelDecls)
    DeclsToIndex.assign(TopLevelDecls->begin(), TopLevelDecls->end());
  else
    DeclsToIndex.assign(AST.getTranslationUnitDecl()->decls().begin(),
                        AST.getTranslationUnitDecl()->decls().end());

  // We only collect refs when indexing main AST.
  // FIXME: this is a hacky way to detect whether we are indexing preamble AST
  // or main AST, we should make it explicitly.
  bool IsIndexMainAST = TopLevelDecls.hasValue();
  if (IsIndexMainAST)
    CollectorOpts.RefFilter = RefKind::All;

  SymbolCollector Collector(std::move(CollectorOpts));
  Collector.setPreprocessor(PP);
  index::indexTopLevelDecls(AST, DeclsToIndex, Collector, IndexOpts);

  const auto &SM = AST.getSourceManager();
  const auto *MainFileEntry = SM.getFileEntryForID(SM.getMainFileID());
  std::string FileName = MainFileEntry ? MainFileEntry->getName() : "";

  auto Syms = Collector.takeSymbols();
  auto Refs = Collector.takeRefs();
  vlog("index {0}AST for {1}: \n"
       "  symbol slab: {2} symbols, {3} bytes\n"
       "  ref slab: {4} symbols, {5} bytes",
       IsIndexMainAST ? "Main" : "Preamble", FileName, Syms.size(),
       Syms.bytes(), Refs.size(), Refs.bytes());
  return {std::move(Syms), std::move(Refs)};
}

FileIndex::FileIndex(std::vector<std::string> URISchemes)
    : URISchemes(std::move(URISchemes)) {
  reset(FSymbols.buildMemIndex());
}

void FileSymbols::update(PathRef Path, std::unique_ptr<SymbolSlab> Symbols,
                         std::unique_ptr<RefSlab> Refs) {
  std::lock_guard<std::mutex> Lock(Mutex);
  if (!Symbols)
    FileToSymbols.erase(Path);
  else
    FileToSymbols[Path] = std::move(Symbols);
  if (!Refs)
    FileToRefs.erase(Path);
  else
    FileToRefs[Path] = std::move(Refs);
}

std::unique_ptr<SymbolIndex> FileSymbols::buildMemIndex() {
  std::vector<std::shared_ptr<SymbolSlab>> SymbolSlabs;
  std::vector<std::shared_ptr<RefSlab>> RefSlabs;
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    for (const auto &FileAndSymbols : FileToSymbols)
      SymbolSlabs.push_back(FileAndSymbols.second);
    for (const auto &FileAndRefs : FileToRefs)
      RefSlabs.push_back(FileAndRefs.second);
  }
  std::vector<const Symbol *> AllSymbols;
  for (const auto &Slab : SymbolSlabs)
    for (const auto &Sym : *Slab)
      AllSymbols.push_back(&Sym);

  std::vector<Ref> RefsStorage; // Contiguous ranges for each SymbolID.
  llvm::DenseMap<SymbolID, ArrayRef<Ref>> AllRefs;
  {
    llvm::DenseMap<SymbolID, SmallVector<Ref, 4>> MergedRefs;
    size_t Count = 0;
    for (const auto &RefSlab : RefSlabs)
      for (const auto &Sym : *RefSlab) {
        MergedRefs[Sym.first].append(Sym.second.begin(), Sym.second.end());
        Count += Sym.second.size();
      }
    RefsStorage.reserve(Count);
    AllRefs.reserve(MergedRefs.size());
    for (auto &Sym : MergedRefs) {
      auto &SymRefs = Sym.second;
      // Sorting isn't required, but yields more stable results over rebuilds.
      std::sort(SymRefs.begin(), SymRefs.end());
      std::copy(SymRefs.begin(), SymRefs.end(), back_inserter(RefsStorage));
      AllRefs.try_emplace(
          Sym.first,
          ArrayRef<Ref>(&RefsStorage[RefsStorage.size() - SymRefs.size()],
                        SymRefs.size()));
    }
  }

  // Index must keep the slabs and contiguous ranges alive.
  return llvm::make_unique<MemIndex>(
      llvm::make_pointee_range(AllSymbols), std::move(AllRefs),
      std::make_tuple(std::move(SymbolSlabs), std::move(RefSlabs),
                      std::move(RefsStorage)));
}

void FileIndex::update(PathRef Path, ASTContext *AST,
                       std::shared_ptr<Preprocessor> PP,
                       llvm::Optional<llvm::ArrayRef<Decl *>> TopLevelDecls) {
  if (!AST) {
    FSymbols.update(Path, nullptr, nullptr);
  } else {
    assert(PP);
    auto Contents = indexAST(*AST, PP, TopLevelDecls, URISchemes);
    FSymbols.update(Path,
                    llvm::make_unique<SymbolSlab>(std::move(Contents.first)),
                    llvm::make_unique<RefSlab>(std::move(Contents.second)));
  }
  reset(FSymbols.buildMemIndex());
}

} // namespace clangd
} // namespace clang
