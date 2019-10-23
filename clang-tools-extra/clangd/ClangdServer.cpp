//===--- ClangdServer.cpp - Main clangd server code --------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------===//

#include "ClangdServer.h"
#include "CodeComplete.h"
#include "FindSymbols.h"
#include "Format.h"
#include "FormattedString.h"
#include "HeaderSourceSwitch.h"
#include "Headers.h"
#include "Logger.h"
#include "ParsedAST.h"
#include "Preamble.h"
#include "Protocol.h"
#include "SemanticHighlighting.h"
#include "SemanticSelection.h"
#include "SourceCode.h"
#include "TUScheduler.h"
#include "Trace.h"
#include "XRefs.h"
#include "index/CanonicalIncludes.h"
#include "index/FileIndex.h"
#include "index/Merge.h"
#include "refactor/Rename.h"
#include "refactor/Tweak.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <future>
#include <memory>
#include <mutex>
#include <type_traits>

namespace clang {
namespace clangd {
namespace {

// Update the FileIndex with new ASTs and plumb the diagnostics responses.
struct UpdateIndexCallbacks : public ParsingCallbacks {
  UpdateIndexCallbacks(FileIndex *FIndex, DiagnosticsConsumer &DiagConsumer,
                       bool SemanticHighlighting)
      : FIndex(FIndex), DiagConsumer(DiagConsumer),
        SemanticHighlighting(SemanticHighlighting) {}

  void onPreambleAST(PathRef Path, ASTContext &Ctx,
                     std::shared_ptr<clang::Preprocessor> PP,
                     const CanonicalIncludes &CanonIncludes) override {
    if (FIndex)
      FIndex->updatePreamble(Path, Ctx, std::move(PP), CanonIncludes);
  }

  void onMainAST(PathRef Path, ParsedAST &AST, PublishFn Publish) override {
    if (FIndex)
      FIndex->updateMain(Path, AST);

    std::vector<Diag> Diagnostics = AST.getDiagnostics();
    std::vector<HighlightingToken> Highlightings;
    if (SemanticHighlighting)
      Highlightings = getSemanticHighlightings(AST);

    Publish([&]() {
      DiagConsumer.onDiagnosticsReady(Path, std::move(Diagnostics));
      if (SemanticHighlighting)
        DiagConsumer.onHighlightingsReady(Path, std::move(Highlightings));
    });
  }

  void onFailedAST(PathRef Path, std::vector<Diag> Diags,
                   PublishFn Publish) override {
    Publish([&]() { DiagConsumer.onDiagnosticsReady(Path, Diags); });
  }

  void onFileUpdated(PathRef File, const TUStatus &Status) override {
    DiagConsumer.onFileUpdated(File, Status);
  }

private:
  FileIndex *FIndex;
  DiagnosticsConsumer &DiagConsumer;
  bool SemanticHighlighting;
};
} // namespace

ClangdServer::Options ClangdServer::optsForTest() {
  ClangdServer::Options Opts;
  Opts.UpdateDebounce = std::chrono::steady_clock::duration::zero(); // Faster!
  Opts.StorePreamblesInMemory = true;
  Opts.AsyncThreadsCount = 4; // Consistent!
  Opts.SemanticHighlighting = true;
  return Opts;
}

ClangdServer::ClangdServer(const GlobalCompilationDatabase &CDB,
                           const FileSystemProvider &FSProvider,
                           DiagnosticsConsumer &DiagConsumer,
                           const Options &Opts)
    : FSProvider(FSProvider),
      DynamicIdx(Opts.BuildDynamicSymbolIndex
                     ? new FileIndex(Opts.HeavyweightDynamicSymbolIndex)
                     : nullptr),
      GetClangTidyOptions(Opts.GetClangTidyOptions),
      SuggestMissingIncludes(Opts.SuggestMissingIncludes),
      CrossFileRename(Opts.CrossFileRename), TweakFilter(Opts.TweakFilter),
      WorkspaceRoot(Opts.WorkspaceRoot),
      // Pass a callback into `WorkScheduler` to extract symbols from a newly
      // parsed file and rebuild the file index synchronously each time an AST
      // is parsed.
      // FIXME(ioeric): this can be slow and we may be able to index on less
      // critical paths.
      WorkScheduler(
          CDB, Opts.AsyncThreadsCount, Opts.StorePreamblesInMemory,
          std::make_unique<UpdateIndexCallbacks>(DynamicIdx.get(), DiagConsumer,
                                                 Opts.SemanticHighlighting),
          Opts.UpdateDebounce, Opts.RetentionPolicy) {
  // Adds an index to the stack, at higher priority than existing indexes.
  auto AddIndex = [&](SymbolIndex *Idx) {
    if (this->Index != nullptr) {
      MergedIdx.push_back(std::make_unique<MergedIndex>(Idx, this->Index));
      this->Index = MergedIdx.back().get();
    } else {
      this->Index = Idx;
    }
  };
  if (Opts.StaticIndex)
    AddIndex(Opts.StaticIndex);
  if (Opts.BackgroundIndex) {
    BackgroundIdx = std::make_unique<BackgroundIndex>(
        Context::current().clone(), FSProvider, CDB,
        BackgroundIndexStorage::createDiskBackedStorageFactory(
            [&CDB](llvm::StringRef File) { return CDB.getProjectInfo(File); }),
        std::max(Opts.AsyncThreadsCount, 1u));
    AddIndex(BackgroundIdx.get());
  }
  if (DynamicIdx)
    AddIndex(DynamicIdx.get());
}

void ClangdServer::addDocument(PathRef File, llvm::StringRef Contents,
                               WantDiagnostics WantDiags) {
  auto FS = FSProvider.getFileSystem();

  ParseOptions Opts;
  Opts.ClangTidyOpts = tidy::ClangTidyOptions::getDefaults();
  // FIXME: call tidy options builder on the worker thread, it can do IO.
  if (GetClangTidyOptions)
    Opts.ClangTidyOpts = GetClangTidyOptions(*FS, File);
  Opts.SuggestMissingIncludes = SuggestMissingIncludes;

  // Compile command is set asynchronously during update, as it can be slow.
  ParseInputs Inputs;
  Inputs.FS = FS;
  Inputs.Contents = Contents;
  Inputs.Opts = std::move(Opts);
  Inputs.Index = Index;
  bool NewFile = WorkScheduler.update(File, Inputs, WantDiags);
  // If we loaded Foo.h, we want to make sure Foo.cpp is indexed.
  if (NewFile && BackgroundIdx)
    BackgroundIdx->boostRelated(File);
}

void ClangdServer::removeDocument(PathRef File) { WorkScheduler.remove(File); }

llvm::StringRef ClangdServer::getDocument(PathRef File) const {
  return WorkScheduler.getContents(File);
}

void ClangdServer::codeComplete(PathRef File, Position Pos,
                                const clangd::CodeCompleteOptions &Opts,
                                Callback<CodeCompleteResult> CB) {
  // Copy completion options for passing them to async task handler.
  auto CodeCompleteOpts = Opts;
  if (!CodeCompleteOpts.Index) // Respect overridden index.
    CodeCompleteOpts.Index = Index;

  auto Task = [Pos, FS = FSProvider.getFileSystem(), CodeCompleteOpts,
               File = File.str(), CB = std::move(CB),
               this](llvm::Expected<InputsAndPreamble> IP) mutable {
    if (!IP)
      return CB(IP.takeError());
    if (isCancelled())
      return CB(llvm::make_error<CancelledError>());

    llvm::Optional<SpeculativeFuzzyFind> SpecFuzzyFind;
    if (!IP->Preamble) {
      // No speculation in Fallback mode, as it's supposed to be much faster
      // without compiling.
      vlog("Build for file {0} is not ready. Enter fallback mode.", File);
    } else {
      if (CodeCompleteOpts.Index && CodeCompleteOpts.SpeculativeIndexRequest) {
        SpecFuzzyFind.emplace();
        {
          std::lock_guard<std::mutex> Lock(
              CachedCompletionFuzzyFindRequestMutex);
          SpecFuzzyFind->CachedReq =
              CachedCompletionFuzzyFindRequestByFile[File];
        }
      }
    }
    // FIXME(ibiryukov): even if Preamble is non-null, we may want to check
    // both the old and the new version in case only one of them matches.
    CodeCompleteResult Result = clangd::codeComplete(
        File, IP->Command, IP->Preamble, IP->Contents, Pos, FS,
        CodeCompleteOpts, SpecFuzzyFind ? SpecFuzzyFind.getPointer() : nullptr);
    {
      clang::clangd::trace::Span Tracer("Completion results callback");
      CB(std::move(Result));
    }
    if (SpecFuzzyFind && SpecFuzzyFind->NewReq.hasValue()) {
      std::lock_guard<std::mutex> Lock(CachedCompletionFuzzyFindRequestMutex);
      CachedCompletionFuzzyFindRequestByFile[File] =
          SpecFuzzyFind->NewReq.getValue();
    }
    // SpecFuzzyFind is only destroyed after speculative fuzzy find finishes.
    // We don't want `codeComplete` to wait for the async call if it doesn't use
    // the result (e.g. non-index completion, speculation fails), so that `CB`
    // is called as soon as results are available.
  };

  // We use a potentially-stale preamble because latency is critical here.
  WorkScheduler.runWithPreamble(
      "CodeComplete", File,
      (Opts.RunParser == CodeCompleteOptions::AlwaysParse)
          ? TUScheduler::Stale
          : TUScheduler::StaleOrAbsent,
      std::move(Task));
}

void ClangdServer::signatureHelp(PathRef File, Position Pos,
                                 Callback<SignatureHelp> CB) {

  auto Action = [Pos, FS = FSProvider.getFileSystem(), File = File.str(),
                 CB = std::move(CB),
                 this](llvm::Expected<InputsAndPreamble> IP) mutable {
    if (!IP)
      return CB(IP.takeError());

    auto PreambleData = IP->Preamble;
    CB(clangd::signatureHelp(File, IP->Command, PreambleData, IP->Contents, Pos,
                             FS, Index));
  };

  // Unlike code completion, we wait for an up-to-date preamble here.
  // Signature help is often triggered after code completion. If the code
  // completion inserted a header to make the symbol available, then using
  // the old preamble would yield useless results.
  WorkScheduler.runWithPreamble("SignatureHelp", File, TUScheduler::Consistent,
                                std::move(Action));
}

llvm::Expected<tooling::Replacements>
ClangdServer::formatRange(llvm::StringRef Code, PathRef File, Range Rng) {
  llvm::Expected<size_t> Begin = positionToOffset(Code, Rng.start);
  if (!Begin)
    return Begin.takeError();
  llvm::Expected<size_t> End = positionToOffset(Code, Rng.end);
  if (!End)
    return End.takeError();
  return formatCode(Code, File, {tooling::Range(*Begin, *End - *Begin)});
}

llvm::Expected<tooling::Replacements>
ClangdServer::formatFile(llvm::StringRef Code, PathRef File) {
  // Format everything.
  return formatCode(Code, File, {tooling::Range(0, Code.size())});
}

llvm::Expected<std::vector<TextEdit>>
ClangdServer::formatOnType(llvm::StringRef Code, PathRef File, Position Pos,
                           StringRef TriggerText) {
  llvm::Expected<size_t> CursorPos = positionToOffset(Code, Pos);
  if (!CursorPos)
    return CursorPos.takeError();
  auto FS = FSProvider.getFileSystem();
  auto Style = format::getStyle(format::DefaultFormatStyle, File,
                                format::DefaultFallbackStyle, Code, FS.get());
  if (!Style)
    return Style.takeError();

  std::vector<TextEdit> Result;
  for (const tooling::Replacement &R :
       formatIncremental(Code, *CursorPos, TriggerText, *Style))
    Result.push_back(replacementToEdit(Code, R));
  return Result;
}

void ClangdServer::prepareRename(PathRef File, Position Pos,
                                 Callback<llvm::Optional<Range>> CB) {
  auto Action = [Pos, File = File.str(), CB = std::move(CB),
                 this](llvm::Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    auto &AST = InpAST->AST;
    const auto &SM = AST.getSourceManager();
    SourceLocation Loc =
        SM.getMacroArgExpandedLocation(getBeginningOfIdentifier(
            Pos, AST.getSourceManager(), AST.getASTContext().getLangOpts()));
    auto Range = getTokenRange(SM, AST.getASTContext().getLangOpts(), Loc);
    if (!Range)
      return CB(llvm::None); // "rename" is not valid at the position.

    if (CrossFileRename)
      // FIXME: we now assume cross-file rename always succeeds, revisit this.
      return CB(*Range);

    // Performing the local rename isn't substantially more expensive than
    // doing an AST-based check, so we just rename and throw away the results.
    auto Changes = clangd::rename({Pos, "dummy", AST, File, Index,
                                   /*AllowCrossFile=*/false,
                                   /*GetDirtyBuffer=*/nullptr});
    if (!Changes) {
      // LSP says to return null on failure, but that will result in a generic
      // failure message. If we send an LSP error response, clients can surface
      // the message to users (VSCode does).
      return CB(Changes.takeError());
    }
    return CB(*Range);
  };
  WorkScheduler.runWithAST("PrepareRename", File, std::move(Action));
}

void ClangdServer::rename(PathRef File, Position Pos, llvm::StringRef NewName,
                          bool WantFormat, Callback<FileEdits> CB) {
  // A snapshot of all file dirty buffers.
  llvm::StringMap<std::string> Snapshot = WorkScheduler.getAllFileContents();
  auto Action = [File = File.str(), NewName = NewName.str(), Pos, WantFormat,
                 CB = std::move(CB), Snapshot = std::move(Snapshot),
                 this](llvm::Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    auto GetDirtyBuffer =
        [&Snapshot](PathRef AbsPath) -> llvm::Optional<std::string> {
      auto It = Snapshot.find(AbsPath);
      if (It == Snapshot.end())
        return llvm::None;
      return It->second;
    };
    auto Edits = clangd::rename({Pos, NewName, InpAST->AST, File, Index,
                                 CrossFileRename, GetDirtyBuffer});
    if (!Edits)
      return CB(Edits.takeError());

    if (WantFormat) {
      auto Style = getFormatStyleForFile(File, InpAST->Inputs.Contents,
                                         InpAST->Inputs.FS.get());
      llvm::Error Err = llvm::Error::success();
      for (auto &E : *Edits)
        Err =
            llvm::joinErrors(reformatEdit(E.getValue(), Style), std::move(Err));

      if (Err)
        return CB(std::move(Err));
    }
    return CB(std::move(*Edits));
  };
  WorkScheduler.runWithAST("Rename", File, std::move(Action));
}

static llvm::Expected<Tweak::Selection>
tweakSelection(const Range &Sel, const InputsAndAST &AST) {
  auto Begin = positionToOffset(AST.Inputs.Contents, Sel.start);
  if (!Begin)
    return Begin.takeError();
  auto End = positionToOffset(AST.Inputs.Contents, Sel.end);
  if (!End)
    return End.takeError();
  return Tweak::Selection(AST.Inputs.Index, AST.AST, *Begin, *End);
}

void ClangdServer::enumerateTweaks(PathRef File, Range Sel,
                                   Callback<std::vector<TweakRef>> CB) {
  auto Action = [File = File.str(), Sel, CB = std::move(CB),
                 this](Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    auto Selection = tweakSelection(Sel, *InpAST);
    if (!Selection)
      return CB(Selection.takeError());
    std::vector<TweakRef> Res;
    for (auto &T : prepareTweaks(*Selection, TweakFilter))
      Res.push_back({T->id(), T->title(), T->intent()});

    CB(std::move(Res));
  };

  WorkScheduler.runWithAST("EnumerateTweaks", File, std::move(Action));
}

void ClangdServer::applyTweak(PathRef File, Range Sel, StringRef TweakID,
                              Callback<Tweak::Effect> CB) {
  auto Action =
      [File = File.str(), Sel, TweakID = TweakID.str(), CB = std::move(CB),
       FS = FSProvider.getFileSystem()](Expected<InputsAndAST> InpAST) mutable {
        if (!InpAST)
          return CB(InpAST.takeError());
        auto Selection = tweakSelection(Sel, *InpAST);
        if (!Selection)
          return CB(Selection.takeError());
        auto A = prepareTweak(TweakID, *Selection);
        if (!A)
          return CB(A.takeError());
        auto Effect = (*A)->apply(*Selection);
        if (!Effect)
          return CB(Effect.takeError());
        for (auto &It : Effect->ApplyEdits) {
          Edit &E = It.second;
          format::FormatStyle Style =
              getFormatStyleForFile(File, E.InitialCode, FS.get());
          if (llvm::Error Err = reformatEdit(E, Style))
            elog("Failed to format {0}: {1}", It.first(), std::move(Err));
        }
        return CB(std::move(*Effect));
      };
  WorkScheduler.runWithAST("ApplyTweak", File, std::move(Action));
}

void ClangdServer::dumpAST(PathRef File,
                           llvm::unique_function<void(std::string)> Callback) {
  auto Action = [Callback = std::move(Callback)](
                    llvm::Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST) {
      llvm::consumeError(InpAST.takeError());
      return Callback("<no-ast>");
    }
    std::string Result;

    llvm::raw_string_ostream ResultOS(Result);
    clangd::dumpAST(InpAST->AST, ResultOS);
    ResultOS.flush();

    Callback(Result);
  };

  WorkScheduler.runWithAST("DumpAST", File, std::move(Action));
}

void ClangdServer::locateSymbolAt(PathRef File, Position Pos,
                                  Callback<std::vector<LocatedSymbol>> CB) {
  auto Action = [Pos, CB = std::move(CB),
                 this](llvm::Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::locateSymbolAt(InpAST->AST, Pos, Index));
  };

  WorkScheduler.runWithAST("Definitions", File, std::move(Action));
}

void ClangdServer::switchSourceHeader(
    PathRef Path, Callback<llvm::Optional<clangd::Path>> CB) {
  // We want to return the result as fast as possible, stragety is:
  //  1) use the file-only heuristic, it requires some IO but it is much
  //     faster than building AST, but it only works when .h/.cc files are in
  //     the same directory.
  //  2) if 1) fails, we use the AST&Index approach, it is slower but supports
  //     different code layout.
  if (auto CorrespondingFile =
          getCorrespondingHeaderOrSource(Path, FSProvider.getFileSystem()))
    return CB(std::move(CorrespondingFile));
  auto Action = [Path = Path.str(), CB = std::move(CB),
                 this](llvm::Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(getCorrespondingHeaderOrSource(Path, InpAST->AST, Index));
  };
  WorkScheduler.runWithAST("SwitchHeaderSource", Path, std::move(Action));
}

llvm::Expected<tooling::Replacements>
ClangdServer::formatCode(llvm::StringRef Code, PathRef File,
                         llvm::ArrayRef<tooling::Range> Ranges) {
  // Call clang-format.
  format::FormatStyle Style =
      getFormatStyleForFile(File, Code, FSProvider.getFileSystem().get());
  tooling::Replacements IncludeReplaces =
      format::sortIncludes(Style, Code, Ranges, File);
  auto Changed = tooling::applyAllReplacements(Code, IncludeReplaces);
  if (!Changed)
    return Changed.takeError();

  return IncludeReplaces.merge(format::reformat(
      Style, *Changed,
      tooling::calculateRangesAfterReplacements(IncludeReplaces, Ranges),
      File));
}

void ClangdServer::findDocumentHighlights(
    PathRef File, Position Pos, Callback<std::vector<DocumentHighlight>> CB) {
  auto Action =
      [Pos, CB = std::move(CB)](llvm::Expected<InputsAndAST> InpAST) mutable {
        if (!InpAST)
          return CB(InpAST.takeError());
        CB(clangd::findDocumentHighlights(InpAST->AST, Pos));
      };

  WorkScheduler.runWithAST("Highlights", File, std::move(Action));
}

void ClangdServer::findHover(PathRef File, Position Pos,
                             Callback<llvm::Optional<HoverInfo>> CB) {
  auto Action = [File = File.str(), Pos, CB = std::move(CB),
                 this](llvm::Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    format::FormatStyle Style = getFormatStyleForFile(
        File, InpAST->Inputs.Contents, InpAST->Inputs.FS.get());
    CB(clangd::getHover(InpAST->AST, Pos, std::move(Style), Index));
  };

  WorkScheduler.runWithAST("Hover", File, std::move(Action));
}

void ClangdServer::typeHierarchy(PathRef File, Position Pos, int Resolve,
                                 TypeHierarchyDirection Direction,
                                 Callback<Optional<TypeHierarchyItem>> CB) {
  auto Action = [File = File.str(), Pos, Resolve, Direction, CB = std::move(CB),
                 this](Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::getTypeHierarchy(InpAST->AST, Pos, Resolve, Direction, Index,
                                File));
  };

  WorkScheduler.runWithAST("Type Hierarchy", File, std::move(Action));
}

void ClangdServer::resolveTypeHierarchy(
    TypeHierarchyItem Item, int Resolve, TypeHierarchyDirection Direction,
    Callback<llvm::Optional<TypeHierarchyItem>> CB) {
  clangd::resolveTypeHierarchy(Item, Resolve, Direction, Index);
  CB(Item);
}

void ClangdServer::onFileEvent(const DidChangeWatchedFilesParams &Params) {
  // FIXME: Do nothing for now. This will be used for indexing and potentially
  // invalidating other caches.
}

void ClangdServer::workspaceSymbols(
    llvm::StringRef Query, int Limit,
    Callback<std::vector<SymbolInformation>> CB) {
  WorkScheduler.run(
      "getWorkspaceSymbols",
      [Query = Query.str(), Limit, CB = std::move(CB), this]() mutable {
        CB(clangd::getWorkspaceSymbols(Query, Limit, Index,
                                       WorkspaceRoot.getValueOr("")));
      });
}

void ClangdServer::documentSymbols(llvm::StringRef File,
                                   Callback<std::vector<DocumentSymbol>> CB) {
  auto Action =
      [CB = std::move(CB)](llvm::Expected<InputsAndAST> InpAST) mutable {
        if (!InpAST)
          return CB(InpAST.takeError());
        CB(clangd::getDocumentSymbols(InpAST->AST));
      };
  WorkScheduler.runWithAST("documentSymbols", File, std::move(Action));
}

void ClangdServer::findReferences(PathRef File, Position Pos, uint32_t Limit,
                                  Callback<ReferencesResult> CB) {
  auto Action = [Pos, Limit, CB = std::move(CB),
                 this](llvm::Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::findReferences(InpAST->AST, Pos, Limit, Index));
  };

  WorkScheduler.runWithAST("References", File, std::move(Action));
}

void ClangdServer::symbolInfo(PathRef File, Position Pos,
                              Callback<std::vector<SymbolDetails>> CB) {
  auto Action =
      [Pos, CB = std::move(CB)](llvm::Expected<InputsAndAST> InpAST) mutable {
        if (!InpAST)
          return CB(InpAST.takeError());
        CB(clangd::getSymbolInfo(InpAST->AST, Pos));
      };

  WorkScheduler.runWithAST("SymbolInfo", File, std::move(Action));
}

void ClangdServer::semanticRanges(PathRef File, Position Pos,
                                  Callback<std::vector<Range>> CB) {
  auto Action =
      [Pos, CB = std::move(CB)](llvm::Expected<InputsAndAST> InpAST) mutable {
        if (!InpAST)
          return CB(InpAST.takeError());
        CB(clangd::getSemanticRanges(InpAST->AST, Pos));
      };
  WorkScheduler.runWithAST("SemanticRanges", File, std::move(Action));
}

std::vector<std::pair<Path, std::size_t>>
ClangdServer::getUsedBytesPerFile() const {
  return WorkScheduler.getUsedBytesPerFile();
}

LLVM_NODISCARD bool
ClangdServer::blockUntilIdleForTest(llvm::Optional<double> TimeoutSeconds) {
  return WorkScheduler.blockUntilIdle(timeoutSeconds(TimeoutSeconds)) &&
         (!BackgroundIdx ||
          BackgroundIdx->blockUntilIdleForTest(TimeoutSeconds));
}

} // namespace clangd
} // namespace clang
