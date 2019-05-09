//===--- FileIndex.cpp - Indexes for files. ------------------------ C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FileIndex.h"
#include "ClangdUnit.h"
#include "Logger.h"
#include "SymbolCollector.h"
#include "index/CanonicalIncludes.h"
#include "index/Index.h"
#include "index/MemIndex.h"
#include "index/Merge.h"
#include "index/SymbolOrigin.h"
#include "index/dex/Dex.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace clang {
namespace clangd {

static std::pair<SymbolSlab, RefSlab>
indexSymbols(ASTContext &AST, std::shared_ptr<Preprocessor> PP,
             llvm::ArrayRef<Decl *> DeclsToIndex,
             const CanonicalIncludes &Includes, bool IsIndexMainAST) {
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
  index::indexTopLevelDecls(AST, *PP, DeclsToIndex, Collector, IndexOpts);

  const auto &SM = AST.getSourceManager();
  const auto *MainFileEntry = SM.getFileEntryForID(SM.getMainFileID());
  std::string FileName = MainFileEntry ? MainFileEntry->getName() : "";

  auto Syms = Collector.takeSymbols();
  auto Refs = Collector.takeRefs();
  vlog("index AST for {0} (main={1}): \n"
       "  symbol slab: {2} symbols, {3} bytes\n"
       "  ref slab: {4} symbols, {5} refs, {6} bytes",
       FileName, IsIndexMainAST, Syms.size(), Syms.bytes(), Refs.size(),
       Refs.numRefs(), Refs.bytes());
  return {std::move(Syms), std::move(Refs)};
}

std::pair<SymbolSlab, RefSlab> indexMainDecls(ParsedAST &AST) {
  return indexSymbols(AST.getASTContext(), AST.getPreprocessorPtr(),
                      AST.getLocalTopLevelDecls(), AST.getCanonicalIncludes(),
                      /*IsIndexMainAST=*/true);
}

SymbolSlab indexHeaderSymbols(ASTContext &AST, std::shared_ptr<Preprocessor> PP,
                              const CanonicalIncludes &Includes) {
  std::vector<Decl *> DeclsToIndex(
      AST.getTranslationUnitDecl()->decls().begin(),
      AST.getTranslationUnitDecl()->decls().end());
  return indexSymbols(AST, std::move(PP), DeclsToIndex, Includes,
                      /*IsIndexMainAST=*/false)
      .first;
}

void FileSymbols::update(PathRef Path, std::unique_ptr<SymbolSlab> Symbols,
                         std::unique_ptr<RefSlab> Refs, bool CountReferences) {
  std::lock_guard<std::mutex> Lock(Mutex);
  if (!Symbols)
    FileToSymbols.erase(Path);
  else
    FileToSymbols[Path] = std::move(Symbols);
  if (!Refs) {
    FileToRefs.erase(Path);
    return;
  }
  RefSlabAndCountReferences Item;
  Item.CountReferences = CountReferences;
  Item.Slab = std::move(Refs);
  FileToRefs[Path] = std::move(Item);
}

std::unique_ptr<SymbolIndex>
FileSymbols::buildIndex(IndexType Type, DuplicateHandling DuplicateHandle) {
  std::vector<std::shared_ptr<SymbolSlab>> SymbolSlabs;
  std::vector<std::shared_ptr<RefSlab>> RefSlabs;
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
        assert(It != Merged.end() && "Reference to unknown symbol");
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

  size_t StorageSize =
      RefsStorage.size() * sizeof(Ref) + SymsStorage.size() * sizeof(Symbol);
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
                        std::move(RefsStorage), std::move(SymsStorage)),
        StorageSize);
  case IndexType::Heavy:
    return llvm::make_unique<dex::Dex>(
        llvm::make_pointee_range(AllSymbols), std::move(AllRefs),
        std::make_tuple(std::move(SymbolSlabs), std::move(RefSlabs),
                        std::move(RefsStorage), std::move(SymsStorage)),
        StorageSize);
  }
  llvm_unreachable("Unknown clangd::IndexType");
}

FileIndex::FileIndex(bool UseDex)
    : MergedIndex(&MainFileIndex, &PreambleIndex), UseDex(UseDex),
      PreambleIndex(llvm::make_unique<MemIndex>()),
      MainFileIndex(llvm::make_unique<MemIndex>()) {}

void FileIndex::updatePreamble(PathRef Path, ASTContext &AST,
                               std::shared_ptr<Preprocessor> PP,
                               const CanonicalIncludes &Includes) {
  auto Symbols = indexHeaderSymbols(AST, std::move(PP), Includes);
  PreambleSymbols.update(
      Path, llvm::make_unique<SymbolSlab>(std::move(Symbols)),
      llvm::make_unique<RefSlab>(), /*CountReferences=*/false);
  PreambleIndex.reset(
      PreambleSymbols.buildIndex(UseDex ? IndexType::Heavy : IndexType::Light,
                                 DuplicateHandling::PickOne));
}

void FileIndex::updateMain(PathRef Path, ParsedAST &AST) {
  auto Contents = indexMainDecls(AST);
  MainFileSymbols.update(
      Path, llvm::make_unique<SymbolSlab>(std::move(Contents.first)),
      llvm::make_unique<RefSlab>(std::move(Contents.second)),
      /*CountReferences=*/true);
  MainFileIndex.reset(
      MainFileSymbols.buildIndex(IndexType::Light, DuplicateHandling::PickOne));
}

} // namespace clangd
} // namespace clang
