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

std::pair<SymbolSlab, SymbolOccurrenceSlab>
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

  // We only collect occurrences when indexing main AST.
  // FIXME: this is a hacky way to detect whether we are indexing preamble AST
  // or main AST, we should make it explicitly.
  bool IsIndexMainAST = TopLevelDecls.hasValue();
  if (IsIndexMainAST)
    CollectorOpts.OccurrenceFilter = AllOccurrenceKinds;

  SymbolCollector Collector(std::move(CollectorOpts));
  Collector.setPreprocessor(PP);
  index::indexTopLevelDecls(AST, DeclsToIndex, Collector, IndexOpts);

  const auto &SM = AST.getSourceManager();
  const auto *MainFileEntry = SM.getFileEntryForID(SM.getMainFileID());
  std::string FileName = MainFileEntry ? MainFileEntry->getName() : "";

  auto Syms = Collector.takeSymbols();
  auto Occurrences = Collector.takeOccurrences();
  vlog("index {0}AST for {1}: \n"
       "  symbol slab: {2} symbols, {3} bytes\n"
       "  occurrence slab: {4} symbols, {5} bytes",
       IsIndexMainAST ? "Main" : "Preamble", FileName, Syms.size(),
       Syms.bytes(), Occurrences.size(), Occurrences.bytes());
  return {std::move(Syms), std::move(Occurrences)};
}

FileIndex::FileIndex(std::vector<std::string> URISchemes)
    : URISchemes(std::move(URISchemes)) {
  reset(FSymbols.buildMemIndex());
}

void FileSymbols::update(PathRef Path, std::unique_ptr<SymbolSlab> Slab,
                         std::unique_ptr<SymbolOccurrenceSlab> Occurrences) {
  std::lock_guard<std::mutex> Lock(Mutex);
  if (!Slab)
    FileToSlabs.erase(Path);
  else
    FileToSlabs[Path] = std::move(Slab);
  if (!Occurrences)
    FileToOccurrenceSlabs.erase(Path);
  else
    FileToOccurrenceSlabs[Path] = std::move(Occurrences);
}

std::unique_ptr<SymbolIndex> FileSymbols::buildMemIndex() {
  std::vector<std::shared_ptr<SymbolSlab>> Slabs;
  std::vector<std::shared_ptr<SymbolOccurrenceSlab>> OccurrenceSlabs;
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    for (const auto &FileAndSlab : FileToSlabs)
      Slabs.push_back(FileAndSlab.second);
    for (const auto &FileAndOccurrenceSlab : FileToOccurrenceSlabs)
      OccurrenceSlabs.push_back(FileAndOccurrenceSlab.second);
  }
  std::vector<const Symbol *> AllSymbols;
  for (const auto &Slab : Slabs)
    for (const auto &Sym : *Slab)
      AllSymbols.push_back(&Sym);
  MemIndex::OccurrenceMap AllOccurrences;
  for (const auto &OccurrenceSlab : OccurrenceSlabs)
    for (const auto &Sym : *OccurrenceSlab) {
      auto &Entry = AllOccurrences[Sym.first];
      for (const auto &Occ : Sym.second)
        Entry.push_back(&Occ);
    }

  // Index must keep the slabs alive.
  return llvm::make_unique<MemIndex>(
      llvm::make_pointee_range(AllSymbols), std::move(AllOccurrences),
      std::make_pair(std::move(Slabs), std::move(OccurrenceSlabs)));
}

void FileIndex::update(PathRef Path, ASTContext *AST,
                       std::shared_ptr<Preprocessor> PP,
                       llvm::Optional<llvm::ArrayRef<Decl *>> TopLevelDecls) {
  if (!AST) {
    FSymbols.update(Path, nullptr, nullptr);
  } else {
    assert(PP);
    auto Slab = llvm::make_unique<SymbolSlab>();
    auto OccurrenceSlab = llvm::make_unique<SymbolOccurrenceSlab>();
    std::tie(*Slab, *OccurrenceSlab) =
        indexAST(*AST, PP, TopLevelDecls, URISchemes);
    FSymbols.update(Path, std::move(Slab), std::move(OccurrenceSlab));
  }
  reset(FSymbols.buildMemIndex());
}

} // namespace clangd
} // namespace clang
