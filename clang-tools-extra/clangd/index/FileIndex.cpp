//===--- FileIndex.cpp - Indexes for files. ------------------------ C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FileIndex.h"
#include "CollectMacros.h"
#include "ParsedAST.h"
#include "index/CanonicalIncludes.h"
#include "index/Index.h"
#include "index/MemIndex.h"
#include "index/Merge.h"
#include "index/Ref.h"
#include "index/Relation.h"
#include "index/Serialization.h"
#include "index/Symbol.h"
#include "index/SymbolCollector.h"
#include "index/SymbolID.h"
#include "index/SymbolOrigin.h"
#include "index/dex/Dex.h"
#include "support/Logger.h"
#include "support/MemoryTree.h"
#include "support/Path.h"
#include "clang/AST/ASTContext.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Index/IndexingOptions.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <algorithm>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

namespace clang {
namespace clangd {
namespace {

SlabTuple indexSymbols(ASTContext &AST, Preprocessor &PP,
                       llvm::ArrayRef<Decl *> DeclsToIndex,
                       const MainFileMacros *MacroRefsToIndex,
                       const CanonicalIncludes &Includes, bool IsIndexMainAST,
                       llvm::StringRef Version, bool CollectMainFileRefs) {
  SymbolCollector::Options CollectorOpts;
  CollectorOpts.CollectIncludePath = true;
  CollectorOpts.Includes = &Includes;
  CollectorOpts.CountReferences = false;
  CollectorOpts.Origin =
      IsIndexMainAST ? SymbolOrigin::Open : SymbolOrigin::Preamble;
  CollectorOpts.CollectMainFileRefs = CollectMainFileRefs;
  // We want stdlib implementation details in the index only if we've opened the
  // file in question. This does means xrefs won't work, though.
  CollectorOpts.CollectReserved = IsIndexMainAST;

  index::IndexingOptions IndexOpts;
  // We only need declarations, because we don't count references.
  IndexOpts.SystemSymbolFilter =
      index::IndexingOptions::SystemSymbolFilterKind::DeclarationsOnly;
  // We index function-local classes and its member functions only.
  IndexOpts.IndexFunctionLocals = true;
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
  index::indexTopLevelDecls(AST, PP, DeclsToIndex, Collector, IndexOpts);
  if (MacroRefsToIndex)
    Collector.handleMacros(*MacroRefsToIndex);

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

// We keep only the node "U" and its edges. Any node other than "U" will be
// empty in the resultant graph.
IncludeGraph getSubGraph(llvm::StringRef URI, const IncludeGraph &FullGraph) {
  IncludeGraph IG;

  auto Entry = IG.try_emplace(URI).first;
  auto &Node = Entry->getValue();
  Node = FullGraph.lookup(Entry->getKey());
  Node.URI = Entry->getKey();

  // URIs inside nodes must point into the keys of the same IncludeGraph.
  for (auto &Include : Node.DirectIncludes) {
    auto I = IG.try_emplace(Include).first;
    I->getValue().URI = I->getKey();
    Include = I->getKey();
  }
  return IG;
}
} // namespace

FileShardedIndex::FileShardedIndex(IndexFileIn Input)
    : Index(std::move(Input)) {
  // Used to build RelationSlabs.
  llvm::DenseMap<SymbolID, FileShard *> SymbolIDToFile;

  // Attribute each Symbol to both their declaration and definition locations.
  if (Index.Symbols) {
    for (const auto &S : *Index.Symbols) {
      auto It = Shards.try_emplace(S.CanonicalDeclaration.FileURI);
      It.first->getValue().Symbols.insert(&S);
      SymbolIDToFile[S.ID] = &It.first->getValue();
      // Only bother if definition file is different than declaration file.
      if (S.Definition &&
          S.Definition.FileURI != S.CanonicalDeclaration.FileURI) {
        auto It = Shards.try_emplace(S.Definition.FileURI);
        It.first->getValue().Symbols.insert(&S);
      }
    }
  }
  // Attribute references into each file they occured in.
  if (Index.Refs) {
    for (const auto &SymRefs : *Index.Refs) {
      for (const auto &R : SymRefs.second) {
        const auto It = Shards.try_emplace(R.Location.FileURI);
        It.first->getValue().Refs.insert(&R);
        RefToSymID[&R] = SymRefs.first;
      }
    }
  }
  // The Subject and/or Object shards might be part of multiple TUs. In
  // such cases there will be a race and the last TU to write the shard
  // will win and all the other relations will be lost. To avoid this,
  // we store relations in both shards. A race might still happen if the
  // same translation unit produces different relations under different
  // configurations, but that's something clangd doesn't handle in general.
  if (Index.Relations) {
    for (const auto &R : *Index.Relations) {
      // FIXME: RelationSlab shouldn't contain dangling relations.
      FileShard *SubjectFile = SymbolIDToFile.lookup(R.Subject);
      FileShard *ObjectFile = SymbolIDToFile.lookup(R.Object);
      if (SubjectFile)
        SubjectFile->Relations.insert(&R);
      if (ObjectFile && ObjectFile != SubjectFile)
        ObjectFile->Relations.insert(&R);
    }
  }
  // Store only the direct includes of a file in a shard.
  if (Index.Sources) {
    const auto &FullGraph = *Index.Sources;
    for (const auto &It : FullGraph) {
      auto ShardIt = Shards.try_emplace(It.first());
      ShardIt.first->getValue().IG = getSubGraph(It.first(), FullGraph);
    }
  }
}
std::vector<llvm::StringRef> FileShardedIndex::getAllSources() const {
  // It should be enough to construct a vector with {Shards.keys().begin(),
  // Shards.keys().end()} but MSVC fails to compile that.
  std::vector<PathRef> Result;
  Result.reserve(Shards.size());
  for (auto Key : Shards.keys())
    Result.push_back(Key);
  return Result;
}

llvm::Optional<IndexFileIn>
FileShardedIndex::getShard(llvm::StringRef Uri) const {
  auto It = Shards.find(Uri);
  if (It == Shards.end())
    return llvm::None;

  IndexFileIn IF;
  IF.Sources = It->getValue().IG;
  IF.Cmd = Index.Cmd;

  SymbolSlab::Builder SymB;
  for (const auto *S : It->getValue().Symbols)
    SymB.insert(*S);
  IF.Symbols = std::move(SymB).build();

  RefSlab::Builder RefB;
  for (const auto *Ref : It->getValue().Refs) {
    auto SID = RefToSymID.lookup(Ref);
    RefB.insert(SID, *Ref);
  }
  IF.Refs = std::move(RefB).build();

  RelationSlab::Builder RelB;
  for (const auto *Rel : It->getValue().Relations) {
    RelB.insert(*Rel);
  }
  IF.Relations = std::move(RelB).build();
  // Explicit move here is needed by some compilers.
  return std::move(IF);
}

SlabTuple indexMainDecls(ParsedAST &AST) {
  return indexSymbols(
      AST.getASTContext(), AST.getPreprocessor(), AST.getLocalTopLevelDecls(),
      &AST.getMacros(), AST.getCanonicalIncludes(),
      /*IsIndexMainAST=*/true, AST.version(), /*CollectMainFileRefs=*/true);
}

SlabTuple indexHeaderSymbols(llvm::StringRef Version, ASTContext &AST,
                             Preprocessor &PP,
                             const CanonicalIncludes &Includes) {
  std::vector<Decl *> DeclsToIndex(
      AST.getTranslationUnitDecl()->decls().begin(),
      AST.getTranslationUnitDecl()->decls().end());
  return indexSymbols(AST, PP, DeclsToIndex,
                      /*MainFileMacros=*/nullptr, Includes,
                      /*IsIndexMainAST=*/false, Version,
                      /*CollectMainFileRefs=*/false);
}

FileSymbols::FileSymbols(IndexContents IdxContents)
    : IdxContents(IdxContents) {}

void FileSymbols::update(llvm::StringRef Key,
                         std::unique_ptr<SymbolSlab> Symbols,
                         std::unique_ptr<RefSlab> Refs,
                         std::unique_ptr<RelationSlab> Relations,
                         bool CountReferences) {
  std::lock_guard<std::mutex> Lock(Mutex);
  ++Version;
  if (!Symbols)
    SymbolsSnapshot.erase(Key);
  else
    SymbolsSnapshot[Key] = std::move(Symbols);
  if (!Refs) {
    RefsSnapshot.erase(Key);
  } else {
    RefSlabAndCountReferences Item;
    Item.CountReferences = CountReferences;
    Item.Slab = std::move(Refs);
    RefsSnapshot[Key] = std::move(Item);
  }
  if (!Relations)
    RelationsSnapshot.erase(Key);
  else
    RelationsSnapshot[Key] = std::move(Relations);
}

std::unique_ptr<SymbolIndex>
FileSymbols::buildIndex(IndexType Type, DuplicateHandling DuplicateHandle,
                        size_t *Version) {
  std::vector<std::shared_ptr<SymbolSlab>> SymbolSlabs;
  std::vector<std::shared_ptr<RefSlab>> RefSlabs;
  std::vector<std::shared_ptr<RelationSlab>> RelationSlabs;
  llvm::StringSet<> Files;
  std::vector<RefSlab *> MainFileRefs;
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    for (const auto &FileAndSymbols : SymbolsSnapshot) {
      SymbolSlabs.push_back(FileAndSymbols.second);
      Files.insert(FileAndSymbols.first());
    }
    for (const auto &FileAndRefs : RefsSnapshot) {
      RefSlabs.push_back(FileAndRefs.second.Slab);
      Files.insert(FileAndRefs.first());
      if (FileAndRefs.second.CountReferences)
        MainFileRefs.push_back(RefSlabs.back().get());
    }
    for (const auto &FileAndRelations : RelationsSnapshot) {
      Files.insert(FileAndRelations.first());
      RelationSlabs.push_back(FileAndRelations.second);
    }

    if (Version)
      *Version = this->Version;
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
  // Sort relations and remove duplicates that could arise due to
  // relations being stored in both the shards containing their
  // subject and object.
  llvm::sort(AllRelations);
  AllRelations.erase(std::unique(AllRelations.begin(), AllRelations.end()),
                     AllRelations.end());

  size_t StorageSize =
      RefsStorage.size() * sizeof(Ref) + SymsStorage.size() * sizeof(Symbol);
  for (const auto &Slab : SymbolSlabs)
    StorageSize += Slab->bytes();
  for (const auto &RefSlab : RefSlabs)
    StorageSize += RefSlab->bytes();

  // Index must keep the slabs and contiguous ranges alive.
  switch (Type) {
  case IndexType::Light:
    return std::make_unique<MemIndex>(
        llvm::make_pointee_range(AllSymbols), std::move(AllRefs),
        std::move(AllRelations), std::move(Files), IdxContents,
        std::make_tuple(std::move(SymbolSlabs), std::move(RefSlabs),
                        std::move(RefsStorage), std::move(SymsStorage)),
        StorageSize);
  case IndexType::Heavy:
    return std::make_unique<dex::Dex>(
        llvm::make_pointee_range(AllSymbols), std::move(AllRefs),
        std::move(AllRelations), std::move(Files), IdxContents,
        std::make_tuple(std::move(SymbolSlabs), std::move(RefSlabs),
                        std::move(RefsStorage), std::move(SymsStorage)),
        StorageSize);
  }
  llvm_unreachable("Unknown clangd::IndexType");
}

void FileSymbols::profile(MemoryTree &MT) const {
  std::lock_guard<std::mutex> Lock(Mutex);
  for (const auto &SymSlab : SymbolsSnapshot) {
    MT.detail(SymSlab.first())
        .child("symbols")
        .addUsage(SymSlab.second->bytes());
  }
  for (const auto &RefSlab : RefsSnapshot) {
    MT.detail(RefSlab.first())
        .child("references")
        .addUsage(RefSlab.second.Slab->bytes());
  }
  for (const auto &RelSlab : RelationsSnapshot) {
    MT.detail(RelSlab.first())
        .child("relations")
        .addUsage(RelSlab.second->bytes());
  }
}

FileIndex::FileIndex()
    : MergedIndex(&MainFileIndex, &PreambleIndex),
      PreambleSymbols(IndexContents::Symbols | IndexContents::Relations),
      PreambleIndex(std::make_unique<MemIndex>()),
      MainFileSymbols(IndexContents::All),
      MainFileIndex(std::make_unique<MemIndex>()) {}

void FileIndex::updatePreamble(PathRef Path, llvm::StringRef Version,
                               ASTContext &AST, Preprocessor &PP,
                               const CanonicalIncludes &Includes) {
  IndexFileIn IF;
  std::tie(IF.Symbols, std::ignore, IF.Relations) =
      indexHeaderSymbols(Version, AST, PP, Includes);
  FileShardedIndex ShardedIndex(std::move(IF));
  for (auto Uri : ShardedIndex.getAllSources()) {
    auto IF = ShardedIndex.getShard(Uri);
    // We are using the key received from ShardedIndex, so it should always
    // exist.
    assert(IF);
    PreambleSymbols.update(
        Uri, std::make_unique<SymbolSlab>(std::move(*IF->Symbols)),
        std::make_unique<RefSlab>(),
        std::make_unique<RelationSlab>(std::move(*IF->Relations)),
        /*CountReferences=*/false);
  }
  size_t IndexVersion = 0;
  auto NewIndex = PreambleSymbols.buildIndex(
      IndexType::Heavy, DuplicateHandling::PickOne, &IndexVersion);
  {
    std::lock_guard<std::mutex> Lock(UpdateIndexMu);
    if (IndexVersion <= PreambleIndexVersion) {
      // We lost the race, some other thread built a later version.
      return;
    }
    PreambleIndexVersion = IndexVersion;
    PreambleIndex.reset(std::move(NewIndex));
    vlog(
        "Build dynamic index for header symbols with estimated memory usage of "
        "{0} bytes",
        PreambleIndex.estimateMemoryUsage());
  }
}

void FileIndex::updateMain(PathRef Path, ParsedAST &AST) {
  auto Contents = indexMainDecls(AST);
  MainFileSymbols.update(
      URI::create(Path).toString(),
      std::make_unique<SymbolSlab>(std::move(std::get<0>(Contents))),
      std::make_unique<RefSlab>(std::move(std::get<1>(Contents))),
      std::make_unique<RelationSlab>(std::move(std::get<2>(Contents))),
      /*CountReferences=*/true);
  size_t IndexVersion = 0;
  auto NewIndex = MainFileSymbols.buildIndex(
      IndexType::Light, DuplicateHandling::Merge, &IndexVersion);
  {
    std::lock_guard<std::mutex> Lock(UpdateIndexMu);
    if (IndexVersion <= MainIndexVersion) {
      // We lost the race, some other thread built a later version.
      return;
    }
    MainIndexVersion = IndexVersion;
    MainFileIndex.reset(std::move(NewIndex));
    vlog(
        "Build dynamic index for main-file symbols with estimated memory usage "
        "of {0} bytes",
        MainFileIndex.estimateMemoryUsage());
  }
}

void FileIndex::profile(MemoryTree &MT) const {
  PreambleSymbols.profile(MT.child("preamble").child("slabs"));
  MT.child("preamble")
      .child("index")
      .addUsage(PreambleIndex.estimateMemoryUsage());
  MainFileSymbols.profile(MT.child("main_file").child("slabs"));
  MT.child("main_file")
      .child("index")
      .addUsage(MainFileIndex.estimateMemoryUsage());
}
} // namespace clangd
} // namespace clang
