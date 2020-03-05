//===--- FileIndex.cpp - Indexes for files. ------------------------ C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FileIndex.h"
#include "CollectMacros.h"
#include "Logger.h"
#include "ParsedAST.h"
#include "SymbolCollector.h"
#include "index/CanonicalIncludes.h"
#include "index/Index.h"
#include "index/MemIndex.h"
#include "index/Merge.h"
#include "index/SymbolOrigin.h"
#include "index/dex/Dex.h"
#include "clang/AST/ASTContext.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Index/IndexingOptions.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace clang {
namespace clangd {

static SlabTuple indexSymbols(ASTContext &AST, std::shared_ptr<Preprocessor> PP,
                              llvm::ArrayRef<Decl *> DeclsToIndex,
                              const MainFileMacros *MacroRefsToIndex,
                              const CanonicalIncludes &Includes,
                              bool IsIndexMainAST, llvm::StringRef Version) {
  SymbolCollector::Options CollectorOpts;
  CollectorOpts.CollectIncludePath = true;
  CollectorOpts.Includes = &Includes;
  CollectorOpts.CountReferences = false;
  CollectorOpts.Origin = SymbolOrigin::Dynamic;

  index::IndexingOptions IndexOpts;
  // We only need declarations, because we don't count references.
  IndexOpts.SystemSymbolFilter =
      index::IndexingOptions::SystemSymbolFilterKind::DeclarationsOnly;
  IndexOpts.IndexFunctionLocals = false;
  if (IsIndexMainAST) {
    // We only collect refs when indexing main AST.
    CollectorOpts.RefFilter = RefKind::All;
    // Comments for main file can always be obtained from sema, do not store
    // them in the index.
    CollectorOpts.StoreAllDocumentation = false;
  } else {
    IndexOpts.IndexMacrosInPreprocessor = true;
    CollectorOpts.CollectMacro = true;
    CollectorOpts.StoreAllDocumentation = true;
  }

  SymbolCollector Collector(std::move(CollectorOpts));
  Collector.setPreprocessor(PP);
  if (MacroRefsToIndex)
    Collector.handleMacros(*MacroRefsToIndex);
  index::indexTopLevelDecls(AST, *PP, DeclsToIndex, Collector, IndexOpts);

  const auto &SM = AST.getSourceManager();
  const auto *MainFileEntry = SM.getFileEntryForID(SM.getMainFileID());
  std::string FileName =
      std::string(MainFileEntry ? MainFileEntry->getName() : "");

  auto Syms = Collector.takeSymbols();
  auto Refs = Collector.takeRefs();
  auto Relations = Collector.takeRelations();

  vlog("indexed {0} AST for {1} version {2}:\n"
       "  symbol slab: {3} symbols, {4} bytes\n"
       "  ref slab: {5} symbols, {6} refs, {7} bytes\n"
       "  relations slab: {8} relations, {9} bytes",
       IsIndexMainAST ? "file" : "preamble", FileName, Version, Syms.size(),
       Syms.bytes(), Refs.size(), Refs.numRefs(), Refs.bytes(),
       Relations.size(), Relations.bytes());
  return std::make_tuple(std::move(Syms), std::move(Refs),
                         std::move(Relations));
}

SlabTuple indexMainDecls(ParsedAST &AST) {
  return indexSymbols(AST.getASTContext(), AST.getPreprocessorPtr(),
                      AST.getLocalTopLevelDecls(), &AST.getMacros(),
                      AST.getCanonicalIncludes(),
                      /*IsIndexMainAST=*/true, AST.version());
}

SlabTuple indexHeaderSymbols(llvm::StringRef Version, ASTContext &AST,
                             std::shared_ptr<Preprocessor> PP,
                             const CanonicalIncludes &Includes) {
  std::vector<Decl *> DeclsToIndex(
      AST.getTranslationUnitDecl()->decls().begin(),
      AST.getTranslationUnitDecl()->decls().end());
  return indexSymbols(AST, std::move(PP), DeclsToIndex,
                      /*MainFileMacros=*/nullptr, Includes,
                      /*IsIndexMainAST=*/false, Version);
}

void FileSymbols::update(PathRef Path, std::unique_ptr<SymbolSlab> Symbols,
                         std::unique_ptr<RefSlab> Refs,
                         std::unique_ptr<RelationSlab> Relations,
                         bool CountReferences) {
  std::lock_guard<std::mutex> Lock(Mutex);
  if (!Symbols)
    FileToSymbols.erase(Path);
  else
    FileToSymbols[Path] = std::move(Symbols);
  if (!Refs) {
    FileToRefs.erase(Path);
  } else {
    RefSlabAndCountReferences Item;
    Item.CountReferences = CountReferences;
    Item.Slab = std::move(Refs);
    FileToRefs[Path] = std::move(Item);
  }
  if (!Relations)
    FileToRelations.erase(Path);
  else
    FileToRelations[Path] = std::move(Relations);
}

std::unique_ptr<SymbolIndex>
FileSymbols::buildIndex(IndexType Type, DuplicateHandling DuplicateHandle) {
  std::vector<std::shared_ptr<SymbolSlab>> SymbolSlabs;
  std::vector<std::shared_ptr<RefSlab>> RefSlabs;
  std::vector<std::shared_ptr<RelationSlab>> RelationSlabs;
  std::vector<RefSlab *> MainFileRefs;
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    for (const auto &FileAndSymbols : FileToSymbols)
      SymbolSlabs.push_back(FileAndSymbols.second);
    for (const auto &FileAndRefs : FileToRefs) {
      RefSlabs.push_back(FileAndRefs.second.Slab);
      if (FileAndRefs.second.CountReferences)
        MainFileRefs.push_back(RefSlabs.back().get());
    }
    for (const auto &FileAndRelations : FileToRelations)
      RelationSlabs.push_back(FileAndRelations.second);
  }
  std::vector<const Symbol *> AllSymbols;
  std::vector<Symbol> SymsStorage;
  switch (DuplicateHandle) {
  case DuplicateHandling::Merge: {
    llvm::DenseMap<SymbolID, Symbol> Merged;
    for (const auto &Slab : SymbolSlabs) {
      for (const auto &Sym : *Slab) {
        assert(Sym.References == 0 &&
               "Symbol with non-zero references sent to FileSymbols");
        auto I = Merged.try_emplace(Sym.ID, Sym);
        if (!I.second)
          I.first->second = mergeSymbol(I.first->second, Sym);
      }
    }
    for (const RefSlab *Refs : MainFileRefs)
      for (const auto &Sym : *Refs) {
        auto It = Merged.find(Sym.first);
        // This might happen while background-index is still running.
        if (It == Merged.end())
          continue;
        It->getSecond().References += Sym.second.size();
      }
    SymsStorage.reserve(Merged.size());
    for (auto &Sym : Merged) {
      SymsStorage.push_back(std::move(Sym.second));
      AllSymbols.push_back(&SymsStorage.back());
    }
    break;
  }
  case DuplicateHandling::PickOne: {
    llvm::DenseSet<SymbolID> AddedSymbols;
    for (const auto &Slab : SymbolSlabs)
      for (const auto &Sym : *Slab) {
        assert(Sym.References == 0 &&
               "Symbol with non-zero references sent to FileSymbols");
        if (AddedSymbols.insert(Sym.ID).second)
          AllSymbols.push_back(&Sym);
      }
    break;
  }
  }

  std::vector<Ref> RefsStorage; // Contiguous ranges for each SymbolID.
  llvm::DenseMap<SymbolID, llvm::ArrayRef<Ref>> AllRefs;
  {
    llvm::DenseMap<SymbolID, llvm::SmallVector<Ref, 4>> MergedRefs;
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
          llvm::ArrayRef<Ref>(&RefsStorage[RefsStorage.size() - SymRefs.size()],
                              SymRefs.size()));
    }
  }

  std::vector<Relation> AllRelations;
  for (const auto &RelationSlab : RelationSlabs) {
    for (const auto &R : *RelationSlab)
      AllRelations.push_back(R);
  }

  size_t StorageSize =
      RefsStorage.size() * sizeof(Ref) + SymsStorage.size() * sizeof(Symbol);
  for (const auto &Slab : SymbolSlabs)
    StorageSize += Slab->bytes();
  for (const auto &RefSlab : RefSlabs)
    StorageSize += RefSlab->bytes();
  for (const auto &RelationSlab : RelationSlabs)
    StorageSize += RelationSlab->bytes();

  // Index must keep the slabs and contiguous ranges alive.
  switch (Type) {
  case IndexType::Light:
    return std::make_unique<MemIndex>(
        llvm::make_pointee_range(AllSymbols), std::move(AllRefs),
        std::move(AllRelations),
        std::make_tuple(std::move(SymbolSlabs), std::move(RefSlabs),
                        std::move(RefsStorage), std::move(SymsStorage)),
        StorageSize);
  case IndexType::Heavy:
    return std::make_unique<dex::Dex>(
        llvm::make_pointee_range(AllSymbols), std::move(AllRefs),
        std::move(AllRelations),
        std::make_tuple(std::move(SymbolSlabs), std::move(RefSlabs),
                        std::move(RefsStorage), std::move(SymsStorage)),
        StorageSize);
  }
  llvm_unreachable("Unknown clangd::IndexType");
}

FileIndex::FileIndex(bool UseDex)
    : MergedIndex(&MainFileIndex, &PreambleIndex), UseDex(UseDex),
      PreambleIndex(std::make_unique<MemIndex>()),
      MainFileIndex(std::make_unique<MemIndex>()) {}

void FileIndex::updatePreamble(PathRef Path, llvm::StringRef Version,
                               ASTContext &AST,
                               std::shared_ptr<Preprocessor> PP,
                               const CanonicalIncludes &Includes) {
  auto Slabs = indexHeaderSymbols(Version, AST, std::move(PP), Includes);
  PreambleSymbols.update(
      Path, std::make_unique<SymbolSlab>(std::move(std::get<0>(Slabs))),
      std::make_unique<RefSlab>(),
      std::make_unique<RelationSlab>(std::move(std::get<2>(Slabs))),
      /*CountReferences=*/false);
  PreambleIndex.reset(
      PreambleSymbols.buildIndex(UseDex ? IndexType::Heavy : IndexType::Light,
                                 DuplicateHandling::PickOne));
}

void FileIndex::updateMain(PathRef Path, ParsedAST &AST) {
  auto Contents = indexMainDecls(AST);
  MainFileSymbols.update(
      Path, std::make_unique<SymbolSlab>(std::move(std::get<0>(Contents))),
      std::make_unique<RefSlab>(std::move(std::get<1>(Contents))),
      std::make_unique<RelationSlab>(std::move(std::get<2>(Contents))),
      /*CountReferences=*/true);
  MainFileIndex.reset(
      MainFileSymbols.buildIndex(IndexType::Light, DuplicateHandling::Merge));
}

} // namespace clangd
} // namespace clang
