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
    : URISchemes(std::move(URISchemes)) {}

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

std::shared_ptr<MemIndex::OccurrenceMap> FileSymbols::allOccurrences() const {
  // The snapshot manages life time of symbol occurrence slabs and provides
  // pointers to all occurrences in all occurrence slabs.
  struct Snapshot {
    MemIndex::OccurrenceMap Occurrences; // ID => {Occurrence}
    std::vector<std::shared_ptr<SymbolOccurrenceSlab>> KeepAlive;
  };

  auto Snap = std::make_shared<Snapshot>();
  {
    std::lock_guard<std::mutex> Lock(Mutex);

    for (const auto &FileAndSlab : FileToOccurrenceSlabs) {
      Snap->KeepAlive.push_back(FileAndSlab.second);
      for (const auto &IDAndOccurrences : *FileAndSlab.second) {
        auto &Occurrences = Snap->Occurrences[IDAndOccurrences.first];
        for (const auto &Occurrence : IDAndOccurrences.second)
          Occurrences.push_back(&Occurrence);
      }
    }
  }

  return {std::move(Snap), &Snap->Occurrences};
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
    auto IndexResults = indexAST(*AST, PP, TopLevelDecls, URISchemes);
    std::tie(*Slab, *OccurrenceSlab) =
        indexAST(*AST, PP, TopLevelDecls, URISchemes);
    FSymbols.update(Path, std::move(Slab), std::move(OccurrenceSlab));
  }
  auto Symbols = FSymbols.allSymbols();
  Index.build(std::move(Symbols), FSymbols.allOccurrences());
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

void FileIndex::findOccurrences(
    const OccurrencesRequest &Req,
    llvm::function_ref<void(const SymbolOccurrence &)> Callback) const {
  Index.findOccurrences(Req, Callback);
}

size_t FileIndex::estimateMemoryUsage() const {
  return Index.estimateMemoryUsage();
}

} // namespace clangd
} // namespace clang
