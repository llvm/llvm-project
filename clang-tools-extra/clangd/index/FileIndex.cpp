//===--- FileIndex.cpp - Indexes for files. ------------------------ C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FileIndex.h"
#include "ClangdUnit.h"
#include "Logger.h"
#include "SymbolCollector.h"
#include "index/Index.h"
#include "index/Merge.h"
#include "index/dex/Dex.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"
#include <memory>

namespace clang {
namespace clangd {

static std::pair<SymbolSlab, RefSlab>
indexSymbols(ASTContext &AST, std::shared_ptr<Preprocessor> PP,
             llvm::ArrayRef<Decl *> DeclsToIndex, bool IsIndexMainAST,
             llvm::ArrayRef<std::string> URISchemes) {
  SymbolCollector::Options CollectorOpts;
  // FIXME(ioeric): we might also want to collect include headers. We would need
  // to make sure all includes are canonicalized (with CanonicalIncludes), which
  // is not trivial given the current way of collecting symbols: we only have
  // AST at this point, but we also need preprocessor callbacks (e.g.
  // CommentHandler for IWYU pragma) to canonicalize includes.
  CollectorOpts.CollectIncludePath = false;
  CollectorOpts.CountReferences = false;
  CollectorOpts.Origin = SymbolOrigin::Dynamic;
  if (!URISchemes.empty())
    CollectorOpts.URISchemes = URISchemes;

  index::IndexingOptions IndexOpts;
  // We only need declarations, because we don't count references.
  IndexOpts.SystemSymbolFilter =
      index::IndexingOptions::SystemSymbolFilterKind::DeclarationsOnly;
  IndexOpts.IndexFunctionLocals = false;
  if (IsIndexMainAST) {
    // We only collect refs when indexing main AST.
    CollectorOpts.RefFilter = RefKind::All;
  }else {
    IndexOpts.IndexMacrosInPreprocessor = true;
    CollectorOpts.CollectMacro = true;
  }

  SymbolCollector Collector(std::move(CollectorOpts));
  Collector.setPreprocessor(PP);
  index::indexTopLevelDecls(AST, *PP, DeclsToIndex, Collector, IndexOpts);

  const auto &SM = AST.getSourceManager();
  const auto *MainFileEntry = SM.getFileEntryForID(SM.getMainFileID());
  std::string FileName = MainFileEntry ? MainFileEntry->getName() : "";

  auto Syms = Collector.takeSymbols();
  auto Refs = Collector.takeRefs();
  vlog("index AST for {0} (main={1}): \n"
       "  symbol slab: {2} symbols, {3} bytes\n"
       "  ref slab: {4} symbols, {5} bytes",
       FileName, IsIndexMainAST, Syms.size(), Syms.bytes(), Refs.size(),
       Refs.bytes());
  return {std::move(Syms), std::move(Refs)};
}

std::pair<SymbolSlab, RefSlab>
indexMainDecls(ParsedAST &AST, llvm::ArrayRef<std::string> URISchemes) {
  return indexSymbols(AST.getASTContext(), AST.getPreprocessorPtr(),
                      AST.getLocalTopLevelDecls(),
                      /*IsIndexMainAST=*/true, URISchemes);
}

SymbolSlab indexHeaderSymbols(ASTContext &AST, std::shared_ptr<Preprocessor> PP,
                              llvm::ArrayRef<std::string> URISchemes) {
  std::vector<Decl *> DeclsToIndex(
      AST.getTranslationUnitDecl()->decls().begin(),
      AST.getTranslationUnitDecl()->decls().end());
  return indexSymbols(AST, std::move(PP), DeclsToIndex,
                      /*IsIndexMainAST=*/false, URISchemes)
      .first;
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

std::unique_ptr<SymbolIndex>
FileSymbols::buildIndex(IndexType Type, ArrayRef<std::string> URISchemes) {
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
      llvm::sort(SymRefs);
      llvm::copy(SymRefs, back_inserter(RefsStorage));
      AllRefs.try_emplace(
          Sym.first,
          ArrayRef<Ref>(&RefsStorage[RefsStorage.size() - SymRefs.size()],
                        SymRefs.size()));
    }
  }

  size_t StorageSize = RefsStorage.size() * sizeof(Ref);
  for (const auto &Slab : SymbolSlabs)
    StorageSize += Slab->bytes();
  for (const auto &RefSlab : RefSlabs)
    StorageSize += RefSlab->bytes();

  // Index must keep the slabs and contiguous ranges alive.
  switch (Type) {
  case IndexType::Light:
    return llvm::make_unique<MemIndex>(
        llvm::make_pointee_range(AllSymbols), std::move(AllRefs),
        std::make_tuple(std::move(SymbolSlabs), std::move(RefSlabs),
                        std::move(RefsStorage)),
        StorageSize);
  case IndexType::Heavy:
    return llvm::make_unique<dex::Dex>(
        llvm::make_pointee_range(AllSymbols), std::move(AllRefs),
        std::make_tuple(std::move(SymbolSlabs), std::move(RefSlabs),
                        std::move(RefsStorage)),
        StorageSize, std::move(URISchemes));
  }
}

FileIndex::FileIndex(std::vector<std::string> URISchemes, bool UseDex)
    : MergedIndex(&MainFileIndex, &PreambleIndex), UseDex(UseDex),
      URISchemes(std::move(URISchemes)),
      PreambleIndex(llvm::make_unique<MemIndex>()),
      MainFileIndex(llvm::make_unique<MemIndex>()) {}

void FileIndex::updatePreamble(PathRef Path, ASTContext &AST,
                               std::shared_ptr<Preprocessor> PP) {
  auto Symbols = indexHeaderSymbols(AST, std::move(PP), URISchemes);
  PreambleSymbols.update(Path,
                         llvm::make_unique<SymbolSlab>(std::move(Symbols)),
                         llvm::make_unique<RefSlab>());
  PreambleIndex.reset(PreambleSymbols.buildIndex(
      UseDex ? IndexType::Heavy : IndexType::Light, URISchemes));
}

void FileIndex::updateMain(PathRef Path, ParsedAST &AST) {
  auto Contents = indexMainDecls(AST, URISchemes);
  MainFileSymbols.update(
      Path, llvm::make_unique<SymbolSlab>(std::move(Contents.first)),
      llvm::make_unique<RefSlab>(std::move(Contents.second)));
  MainFileIndex.reset(MainFileSymbols.buildIndex(IndexType::Light, URISchemes));
}

} // namespace clangd
} // namespace clang
