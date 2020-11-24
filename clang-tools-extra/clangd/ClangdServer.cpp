//===--- ClangdServer.cpp - Main clangd server code --------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------===//

#include "ClangdServer.h"
#include "CodeComplete.h"
#include "Config.h"
#include "DumpAST.h"
#include "FindSymbols.h"
#include "Format.h"
#include "HeaderSourceSwitch.h"
#include "Headers.h"
#include "ParsedAST.h"
#include "Preamble.h"
#include "Protocol.h"
#include "SemanticHighlighting.h"
#include "SemanticSelection.h"
#include "SourceCode.h"
#include "TUScheduler.h"
#include "XRefs.h"
#include "index/CanonicalIncludes.h"
#include "index/FileIndex.h"
#include "index/Merge.h"
#include "refactor/Rename.h"
#include "refactor/Tweak.h"
#include "support/Logger.h"
#include "support/Markup.h"
#include "support/MemoryTree.h"
#include "support/ThreadsafeFS.h"
#include "support/Trace.h"
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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>

namespace clang {
namespace clangd {
namespace {

// Update the FileIndex with new ASTs and plumb the diagnostics responses.
struct UpdateIndexCallbacks : public ParsingCallbacks {
  UpdateIndexCallbacks(FileIndex *FIndex,
                       ClangdServer::Callbacks *ServerCallbacks,
                       bool TheiaSemanticHighlighting)
      : FIndex(FIndex), ServerCallbacks(ServerCallbacks),
        TheiaSemanticHighlighting(TheiaSemanticHighlighting) {}

  void onPreambleAST(PathRef Path, llvm::StringRef Version, ASTContext &Ctx,
                     std::shared_ptr<clang::Preprocessor> PP,
                     const CanonicalIncludes &CanonIncludes) override {
    if (FIndex)
      FIndex->updatePreamble(Path, Version, Ctx, std::move(PP), CanonIncludes);
  }

  void onMainAST(PathRef Path, ParsedAST &AST, PublishFn Publish) override {
    if (FIndex)
      FIndex->updateMain(Path, AST);

    std::vector<Diag> Diagnostics = AST.getDiagnostics();
    std::vector<HighlightingToken> Highlightings;
    if (TheiaSemanticHighlighting)
      Highlightings = getSemanticHighlightings(AST);

    if (ServerCallbacks)
      Publish([&]() {
        ServerCallbacks->onDiagnosticsReady(Path, AST.version(),
                                            std::move(Diagnostics));
        if (TheiaSemanticHighlighting)
          ServerCallbacks->onHighlightingsReady(Path, AST.version(),
                                                std::move(Highlightings));
      });
  }

  void onFailedAST(PathRef Path, llvm::StringRef Version,
                   std::vector<Diag> Diags, PublishFn Publish) override {
    if (ServerCallbacks)
      Publish(
          [&]() { ServerCallbacks->onDiagnosticsReady(Path, Version, Diags); });
  }

  void onFileUpdated(PathRef File, const TUStatus &Status) override {
    if (ServerCallbacks)
      ServerCallbacks->onFileUpdated(File, Status);
  }

private:
  FileIndex *FIndex;
  ClangdServer::Callbacks *ServerCallbacks;
  bool TheiaSemanticHighlighting;
};

// Set of clang-tidy checks that don't work well in clangd, either due to
// crashes or false positives.
// Returns a clang-tidy filter string: -check1,-check2.
llvm::StringRef getUnusableTidyChecks() {
  static const std::string FalsePositives =
      llvm::join_items(", ",
                       // Check relies on seeing ifndef/define/endif directives,
                       // clangd doesn't replay those when using a preamble.
                       "-llvm-header-guard");
  static const std::string CrashingChecks =
      llvm::join_items(", ",
                       // Check can choke on invalid (intermediate) c++ code,
                       // which is often the case when clangd tries to build an
                       // AST.
                       "-bugprone-use-after-move");
  static const std::string UnusableTidyChecks =
      llvm::join_items(", ", FalsePositives, CrashingChecks);
  return UnusableTidyChecks;
}

// Returns a clang-tidy options string: check1,check2.
llvm::StringRef getDefaultTidyChecks() {
  // These default checks are chosen for:
  //  - low false-positive rate
  //  - providing a lot of value
  //  - being reasonably efficient
  static const std::string DefaultChecks = llvm::join_items(
      ",", "readability-misleading-indentation", "readability-deleted-default",
      "bugprone-integer-division", "bugprone-sizeof-expression",
      "bugprone-suspicious-missing-comma", "bugprone-unused-raii",
      "bugprone-unused-return-value", "misc-unused-using-decls",
      "misc-unused-alias-decls", "misc-definitions-in-headers");
  return DefaultChecks;
}
} // namespace

ClangdServer::Options ClangdServer::optsForTest() {
  ClangdServer::Options Opts;
  Opts.UpdateDebounce = DebouncePolicy::fixed(/*zero*/ {});
  Opts.StorePreamblesInMemory = true;
  Opts.AsyncThreadsCount = 4; // Consistent!
  Opts.TheiaSemanticHighlighting = true;
  Opts.AsyncPreambleBuilds = true;
  return Opts;
}

ClangdServer::Options::operator TUScheduler::Options() const {
  TUScheduler::Options Opts;
  Opts.AsyncThreadsCount = AsyncThreadsCount;
  Opts.RetentionPolicy = RetentionPolicy;
  Opts.StorePreamblesInMemory = StorePreamblesInMemory;
  Opts.UpdateDebounce = UpdateDebounce;
  Opts.AsyncPreambleBuilds = AsyncPreambleBuilds;
  return Opts;
}

ClangdServer::ClangdServer(const GlobalCompilationDatabase &CDB,
                           const ThreadsafeFS &TFS, const Options &Opts,
                           Callbacks *Callbacks)
    : ConfigProvider(Opts.ConfigProvider), TFS(TFS),
      DynamicIdx(Opts.BuildDynamicSymbolIndex
                     ? new FileIndex(Opts.HeavyweightDynamicSymbolIndex,
                                     Opts.CollectMainFileRefs)
                     : nullptr),
      GetClangTidyOptions(Opts.GetClangTidyOptions),
      SuggestMissingIncludes(Opts.SuggestMissingIncludes),
      BuildRecoveryAST(Opts.BuildRecoveryAST),
      PreserveRecoveryASTType(Opts.PreserveRecoveryASTType),
      WorkspaceRoot(Opts.WorkspaceRoot),
      // Pass a callback into `WorkScheduler` to extract symbols from a newly
      // parsed file and rebuild the file index synchronously each time an AST
      // is parsed.
      // FIXME(ioeric): this can be slow and we may be able to index on less
      // critical paths.
      WorkScheduler(
          CDB,
          [&, this] {
            TUScheduler::Options O(Opts);
            O.ContextProvider = [this](PathRef P) {
              return createProcessingContext(P);
            };
            return O;
          }(),
          std::make_unique<UpdateIndexCallbacks>(
              DynamicIdx.get(), Callbacks, Opts.TheiaSemanticHighlighting)) {
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
    BackgroundIndex::Options BGOpts;
    BGOpts.ThreadPoolSize = std::max(Opts.AsyncThreadsCount, 1u);
    BGOpts.OnProgress = [Callbacks](BackgroundQueue::Stats S) {
      if (Callbacks)
        Callbacks->onBackgroundIndexProgress(S);
    };
    BGOpts.ContextProvider = [this](PathRef P) {
      return createProcessingContext(P);
    };
    BGOpts.CollectMainFileRefs = Opts.CollectMainFileRefs;
    BackgroundIdx = std::make_unique<BackgroundIndex>(
        TFS, CDB,
        BackgroundIndexStorage::createDiskBackedStorageFactory(
            [&CDB](llvm::StringRef File) { return CDB.getProjectInfo(File); }),
        std::move(BGOpts));
    AddIndex(BackgroundIdx.get());
  }
  if (DynamicIdx)
    AddIndex(DynamicIdx.get());
}

void ClangdServer::addDocument(PathRef File, llvm::StringRef Contents,
                               llvm::StringRef Version,
                               WantDiagnostics WantDiags, bool ForceRebuild) {
  ParseOptions Opts;
  Opts.ClangTidyOpts = tidy::ClangTidyOptions::getDefaults();
  // FIXME: call tidy options builder on the worker thread, it can do IO.
  if (GetClangTidyOptions)
    Opts.ClangTidyOpts =
        GetClangTidyOptions(*TFS.view(/*CWD=*/llvm::None), File);
  if (Opts.ClangTidyOpts.Checks.hasValue()) {
    // If the set of checks was configured, make sure clangd incompatible ones
    // are disabled.
    Opts.ClangTidyOpts.Checks = llvm::join_items(
        ", ", *Opts.ClangTidyOpts.Checks, getUnusableTidyChecks());
  } else {
    // Otherwise provide a nice set of defaults.
    Opts.ClangTidyOpts.Checks = getDefaultTidyChecks().str();
  }
  Opts.SuggestMissingIncludes = SuggestMissingIncludes;

  // Compile command is set asynchronously during update, as it can be slow.
  ParseInputs Inputs;
  Inputs.TFS = &TFS;
  Inputs.Contents = std::string(Contents);
  Inputs.Version = Version.str();
  Inputs.ForceRebuild = ForceRebuild;
  Inputs.Opts = std::move(Opts);
  Inputs.Index = Index;
  Inputs.Opts.BuildRecoveryAST = BuildRecoveryAST;
  Inputs.Opts.PreserveRecoveryASTType = PreserveRecoveryASTType;
  bool NewFile = WorkScheduler.update(File, Inputs, WantDiags);
  // If we loaded Foo.h, we want to make sure Foo.cpp is indexed.
  if (NewFile && BackgroundIdx)
    BackgroundIdx->boostRelated(File);
}

void ClangdServer::removeDocument(PathRef File) { WorkScheduler.remove(File); }

void ClangdServer::codeComplete(PathRef File, Position Pos,
                                const clangd::CodeCompleteOptions &Opts,
                                Callback<CodeCompleteResult> CB) {
  // Copy completion options for passing them to async task handler.
  auto CodeCompleteOpts = Opts;
  if (!CodeCompleteOpts.Index) // Respect overridden index.
    CodeCompleteOpts.Index = Index;

  auto Task = [Pos, CodeCompleteOpts, File = File.str(), CB = std::move(CB),
               this](llvm::Expected<InputsAndPreamble> IP) mutable {
    if (!IP)
      return CB(IP.takeError());
    if (auto Reason = isCancelled())
      return CB(llvm::make_error<CancelledError>(Reason));

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
    ParseInputs ParseInput{IP->Command, &TFS, IP->Contents.str()};
    ParseInput.Index = Index;
    ParseInput.Opts.BuildRecoveryAST = BuildRecoveryAST;
    ParseInput.Opts.PreserveRecoveryASTType = PreserveRecoveryASTType;

    // FIXME(ibiryukov): even if Preamble is non-null, we may want to check
    // both the old and the new version in case only one of them matches.
    CodeCompleteResult Result = clangd::codeComplete(
        File, Pos, IP->Preamble, ParseInput, CodeCompleteOpts,
        SpecFuzzyFind ? SpecFuzzyFind.getPointer() : nullptr);
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

  auto Action = [Pos, File = File.str(), CB = std::move(CB),
                 this](llvm::Expected<InputsAndPreamble> IP) mutable {
    if (!IP)
      return CB(IP.takeError());

    const auto *PreambleData = IP->Preamble;
    if (!PreambleData)
      return CB(error("Failed to parse includes"));

    ParseInputs ParseInput{IP->Command, &TFS, IP->Contents.str()};
    ParseInput.Index = Index;
    ParseInput.Opts.BuildRecoveryAST = BuildRecoveryAST;
    ParseInput.Opts.PreserveRecoveryASTType = PreserveRecoveryASTType;
    CB(clangd::signatureHelp(File, Pos, *PreambleData, ParseInput));
  };

  // Unlike code completion, we wait for a preamble here.
  WorkScheduler.runWithPreamble("SignatureHelp", File, TUScheduler::Stale,
                                std::move(Action));
}

void ClangdServer::formatRange(PathRef File, llvm::StringRef Code, Range Rng,
                               Callback<tooling::Replacements> CB) {
  llvm::Expected<size_t> Begin = positionToOffset(Code, Rng.start);
  if (!Begin)
    return CB(Begin.takeError());
  llvm::Expected<size_t> End = positionToOffset(Code, Rng.end);
  if (!End)
    return CB(End.takeError());
  formatCode(File, Code, {tooling::Range(*Begin, *End - *Begin)},
             std::move(CB));
}

void ClangdServer::formatFile(PathRef File, llvm::StringRef Code,
                              Callback<tooling::Replacements> CB) {
  // Format everything.
  formatCode(File, Code, {tooling::Range(0, Code.size())}, std::move(CB));
}

void ClangdServer::formatOnType(PathRef File, llvm::StringRef Code,
                                Position Pos, StringRef TriggerText,
                                Callback<std::vector<TextEdit>> CB) {
  llvm::Expected<size_t> CursorPos = positionToOffset(Code, Pos);
  if (!CursorPos)
    return CB(CursorPos.takeError());
  auto Action = [File = File.str(), Code = Code.str(),
                 TriggerText = TriggerText.str(), CursorPos = *CursorPos,
                 CB = std::move(CB), this]() mutable {
    auto Style = format::getStyle(format::DefaultFormatStyle, File,
                                  format::DefaultFallbackStyle, Code,
                                  TFS.view(/*CWD=*/llvm::None).get());
    if (!Style)
      return CB(Style.takeError());

    std::vector<TextEdit> Result;
    for (const tooling::Replacement &R :
         formatIncremental(Code, CursorPos, TriggerText, *Style))
      Result.push_back(replacementToEdit(Code, R));
    return CB(Result);
  };
  WorkScheduler.run("FormatOnType", File, std::move(Action));
}

void ClangdServer::prepareRename(PathRef File, Position Pos,
                                 llvm::Optional<std::string> NewName,
                                 const RenameOptions &RenameOpts,
                                 Callback<RenameResult> CB) {
  auto Action = [Pos, File = File.str(), CB = std::move(CB),
                 NewName = std::move(NewName), RenameOpts,
                 this](llvm::Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    // prepareRename is latency-sensitive:
    //  - for single-file rename, performing rename isn't substantially more
    //    expensive than doing an AST-based check (the index is used to see if
    //    the rename is complete);
    //  - for cross-file rename, we deliberately pass a nullptr index to save
    //    the cost, thus the result may be incomplete as it only contains
    //    main-file occurrences;
    auto Results = clangd::rename(
        {Pos, NewName.getValueOr("__clangd_rename_dummy"), InpAST->AST, File,
         RenameOpts.AllowCrossFile ? nullptr : Index, RenameOpts});
    if (!Results) {
      // LSP says to return null on failure, but that will result in a generic
      // failure message. If we send an LSP error response, clients can surface
      // the message to users (VSCode does).
      return CB(Results.takeError());
    }
    return CB(*Results);
  };
  WorkScheduler.runWithAST("PrepareRename", File, std::move(Action));
}

void ClangdServer::rename(PathRef File, Position Pos, llvm::StringRef NewName,
                          const RenameOptions &Opts,
                          Callback<RenameResult> CB) {
  // A snapshot of all file dirty buffers.
  llvm::StringMap<std::string> Snapshot = WorkScheduler.getAllFileContents();
  auto Action = [File = File.str(), NewName = NewName.str(), Pos, Opts,
                 CB = std::move(CB), Snapshot = std::move(Snapshot),
                 this](llvm::Expected<InputsAndAST> InpAST) mutable {
    // Tracks number of files edited per invocation.
    static constexpr trace::Metric RenameFiles("rename_files",
                                               trace::Metric::Distribution);
    if (!InpAST)
      return CB(InpAST.takeError());
    auto GetDirtyBuffer =
        [&Snapshot](PathRef AbsPath) -> llvm::Optional<std::string> {
      auto It = Snapshot.find(AbsPath);
      if (It == Snapshot.end())
        return llvm::None;
      return It->second;
    };
    auto R = clangd::rename(
        {Pos, NewName, InpAST->AST, File, Index, Opts, GetDirtyBuffer});
    if (!R)
      return CB(R.takeError());

    if (Opts.WantFormat) {
      auto Style = getFormatStyleForFile(File, InpAST->Inputs.Contents,
                                         *InpAST->Inputs.TFS);
      llvm::Error Err = llvm::Error::success();
      for (auto &E : R->GlobalChanges)
        Err =
            llvm::joinErrors(reformatEdit(E.getValue(), Style), std::move(Err));

      if (Err)
        return CB(std::move(Err));
    }
    RenameFiles.record(R->GlobalChanges.size());
    return CB(*R);
  };
  WorkScheduler.runWithAST("Rename", File, std::move(Action));
}

// May generate several candidate selections, due to SelectionTree ambiguity.
// vector of pointers because GCC doesn't like non-copyable Selection.
static llvm::Expected<std::vector<std::unique_ptr<Tweak::Selection>>>
tweakSelection(const Range &Sel, const InputsAndAST &AST) {
  auto Begin = positionToOffset(AST.Inputs.Contents, Sel.start);
  if (!Begin)
    return Begin.takeError();
  auto End = positionToOffset(AST.Inputs.Contents, Sel.end);
  if (!End)
    return End.takeError();
  std::vector<std::unique_ptr<Tweak::Selection>> Result;
  SelectionTree::createEach(
      AST.AST.getASTContext(), AST.AST.getTokens(), *Begin, *End,
      [&](SelectionTree T) {
        Result.push_back(std::make_unique<Tweak::Selection>(
            AST.Inputs.Index, AST.AST, *Begin, *End, std::move(T)));
        return false;
      });
  assert(!Result.empty() && "Expected at least one SelectionTree");
  return std::move(Result);
}

void ClangdServer::enumerateTweaks(
    PathRef File, Range Sel, llvm::unique_function<bool(const Tweak &)> Filter,
    Callback<std::vector<TweakRef>> CB) {
  // Tracks number of times a tweak has been offered.
  static constexpr trace::Metric TweakAvailable(
      "tweak_available", trace::Metric::Counter, "tweak_id");
  auto Action = [File = File.str(), Sel, CB = std::move(CB),
                 Filter =
                     std::move(Filter)](Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    auto Selections = tweakSelection(Sel, *InpAST);
    if (!Selections)
      return CB(Selections.takeError());
    std::vector<TweakRef> Res;
    // Don't allow a tweak to fire more than once across ambiguous selections.
    llvm::DenseSet<llvm::StringRef> PreparedTweaks;
    auto DeduplicatingFilter = [&](const Tweak &T) {
      return Filter(T) && !PreparedTweaks.count(T.id());
    };
    for (const auto &Sel : *Selections) {
      for (auto &T : prepareTweaks(*Sel, DeduplicatingFilter)) {
        Res.push_back({T->id(), T->title(), T->kind()});
        PreparedTweaks.insert(T->id());
        TweakAvailable.record(1, T->id());
      }
    }

    CB(std::move(Res));
  };

  WorkScheduler.runWithAST("EnumerateTweaks", File, std::move(Action),
                           TUScheduler::InvalidateOnUpdate);
}

void ClangdServer::applyTweak(PathRef File, Range Sel, StringRef TweakID,
                              Callback<Tweak::Effect> CB) {
  // Tracks number of times a tweak has been attempted.
  static constexpr trace::Metric TweakAttempt(
      "tweak_attempt", trace::Metric::Counter, "tweak_id");
  // Tracks number of times a tweak has failed to produce edits.
  static constexpr trace::Metric TweakFailed(
      "tweak_failed", trace::Metric::Counter, "tweak_id");
  TweakAttempt.record(1, TweakID);
  auto Action = [File = File.str(), Sel, TweakID = TweakID.str(),
                 CB = std::move(CB),
                 this](Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    auto Selections = tweakSelection(Sel, *InpAST);
    if (!Selections)
      return CB(Selections.takeError());
    llvm::Optional<llvm::Expected<Tweak::Effect>> Effect;
    // Try each selection, take the first one that prepare()s.
    // If they all fail, Effect will hold get the last error.
    for (const auto &Selection : *Selections) {
      auto T = prepareTweak(TweakID, *Selection);
      if (T) {
        Effect = (*T)->apply(*Selection);
        break;
      }
      Effect = T.takeError();
    }
    assert(Effect.hasValue() && "Expected at least one selection");
    if (*Effect) {
      // Tweaks don't apply clang-format, do that centrally here.
      for (auto &It : (*Effect)->ApplyEdits) {
        Edit &E = It.second;
        format::FormatStyle Style =
            getFormatStyleForFile(File, E.InitialCode, TFS);
        if (llvm::Error Err = reformatEdit(E, Style))
          elog("Failed to format {0}: {1}", It.first(), std::move(Err));
      }
    } else {
      TweakFailed.record(1, TweakID);
    }
    return CB(std::move(*Effect));
  };
  WorkScheduler.runWithAST("ApplyTweak", File, std::move(Action));
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
  // We want to return the result as fast as possible, strategy is:
  //  1) use the file-only heuristic, it requires some IO but it is much
  //     faster than building AST, but it only works when .h/.cc files are in
  //     the same directory.
  //  2) if 1) fails, we use the AST&Index approach, it is slower but supports
  //     different code layout.
  if (auto CorrespondingFile = getCorrespondingHeaderOrSource(
          std::string(Path), TFS.view(llvm::None)))
    return CB(std::move(CorrespondingFile));
  auto Action = [Path = Path.str(), CB = std::move(CB),
                 this](llvm::Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(getCorrespondingHeaderOrSource(Path, InpAST->AST, Index));
  };
  WorkScheduler.runWithAST("SwitchHeaderSource", Path, std::move(Action));
}

void ClangdServer::formatCode(PathRef File, llvm::StringRef Code,
                              llvm::ArrayRef<tooling::Range> Ranges,
                              Callback<tooling::Replacements> CB) {
  // Call clang-format.
  auto Action = [File = File.str(), Code = Code.str(), Ranges = Ranges.vec(),
                 CB = std::move(CB), this]() mutable {
    format::FormatStyle Style = getFormatStyleForFile(File, Code, TFS);
    tooling::Replacements IncludeReplaces =
        format::sortIncludes(Style, Code, Ranges, File);
    auto Changed = tooling::applyAllReplacements(Code, IncludeReplaces);
    if (!Changed)
      return CB(Changed.takeError());

    CB(IncludeReplaces.merge(format::reformat(
        Style, *Changed,
        tooling::calculateRangesAfterReplacements(IncludeReplaces, Ranges),
        File)));
  };
  WorkScheduler.run("Format", File, std::move(Action));
}

void ClangdServer::findDocumentHighlights(
    PathRef File, Position Pos, Callback<std::vector<DocumentHighlight>> CB) {
  auto Action =
      [Pos, CB = std::move(CB)](llvm::Expected<InputsAndAST> InpAST) mutable {
        if (!InpAST)
          return CB(InpAST.takeError());
        CB(clangd::findDocumentHighlights(InpAST->AST, Pos));
      };

  WorkScheduler.runWithAST("Highlights", File, std::move(Action),
                           TUScheduler::InvalidateOnUpdate);
}

void ClangdServer::findHover(PathRef File, Position Pos,
                             Callback<llvm::Optional<HoverInfo>> CB) {
  auto Action = [File = File.str(), Pos, CB = std::move(CB),
                 this](llvm::Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    format::FormatStyle Style = getFormatStyleForFile(
        File, InpAST->Inputs.Contents, *InpAST->Inputs.TFS);
    CB(clangd::getHover(InpAST->AST, Pos, std::move(Style), Index));
  };

  WorkScheduler.runWithAST("Hover", File, std::move(Action),
                           TUScheduler::InvalidateOnUpdate);
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
  WorkScheduler.run(
      "Resolve Type Hierarchy", "", [=, CB = std::move(CB)]() mutable {
        clangd::resolveTypeHierarchy(Item, Resolve, Direction, Index);
        CB(Item);
      });
}

void ClangdServer::prepareCallHierarchy(
    PathRef File, Position Pos, Callback<std::vector<CallHierarchyItem>> CB) {
  auto Action = [File = File.str(), Pos,
                 CB = std::move(CB)](Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::prepareCallHierarchy(InpAST->AST, Pos, File));
  };
  WorkScheduler.runWithAST("Call Hierarchy", File, std::move(Action));
}

void ClangdServer::incomingCalls(
    const CallHierarchyItem &Item,
    Callback<std::vector<CallHierarchyIncomingCall>> CB) {
  WorkScheduler.run("Incoming Calls", "",
                    [CB = std::move(CB), Item, this]() mutable {
                      CB(clangd::incomingCalls(Item, Index));
                    });
}

void ClangdServer::onFileEvent(const DidChangeWatchedFilesParams &Params) {
  // FIXME: Do nothing for now. This will be used for indexing and potentially
  // invalidating other caches.
}

void ClangdServer::workspaceSymbols(
    llvm::StringRef Query, int Limit,
    Callback<std::vector<SymbolInformation>> CB) {
  WorkScheduler.run(
      "getWorkspaceSymbols", /*Path=*/"",
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
  WorkScheduler.runWithAST("documentSymbols", File, std::move(Action),
                           TUScheduler::InvalidateOnUpdate);
}

void ClangdServer::foldingRanges(llvm::StringRef File,
                                 Callback<std::vector<FoldingRange>> CB) {
  auto Action =
      [CB = std::move(CB)](llvm::Expected<InputsAndAST> InpAST) mutable {
        if (!InpAST)
          return CB(InpAST.takeError());
        CB(clangd::getFoldingRanges(InpAST->AST));
      };
  WorkScheduler.runWithAST("foldingRanges", File, std::move(Action),
                           TUScheduler::InvalidateOnUpdate);
}

void ClangdServer::findImplementations(
    PathRef File, Position Pos, Callback<std::vector<LocatedSymbol>> CB) {
  auto Action = [Pos, CB = std::move(CB),
                 this](llvm::Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::findImplementations(InpAST->AST, Pos, Index));
  };

  WorkScheduler.runWithAST("Implementations", File, std::move(Action));
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

void ClangdServer::semanticRanges(PathRef File,
                                  const std::vector<Position> &Positions,
                                  Callback<std::vector<SelectionRange>> CB) {
  auto Action = [Positions, CB = std::move(CB)](
                    llvm::Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    std::vector<SelectionRange> Result;
    for (const auto &Pos : Positions) {
      if (auto Range = clangd::getSemanticRanges(InpAST->AST, Pos))
        Result.push_back(std::move(*Range));
      else
        return CB(Range.takeError());
    }
    CB(std::move(Result));
  };
  WorkScheduler.runWithAST("SemanticRanges", File, std::move(Action));
}

void ClangdServer::documentLinks(PathRef File,
                                 Callback<std::vector<DocumentLink>> CB) {
  auto Action =
      [CB = std::move(CB)](llvm::Expected<InputsAndAST> InpAST) mutable {
        if (!InpAST)
          return CB(InpAST.takeError());
        CB(clangd::getDocumentLinks(InpAST->AST));
      };
  WorkScheduler.runWithAST("DocumentLinks", File, std::move(Action),
                           TUScheduler::InvalidateOnUpdate);
}

void ClangdServer::semanticHighlights(
    PathRef File, Callback<std::vector<HighlightingToken>> CB) {
  auto Action =
      [CB = std::move(CB)](llvm::Expected<InputsAndAST> InpAST) mutable {
        if (!InpAST)
          return CB(InpAST.takeError());
        CB(clangd::getSemanticHighlightings(InpAST->AST));
      };
  WorkScheduler.runWithAST("SemanticHighlights", File, std::move(Action),
                           TUScheduler::InvalidateOnUpdate);
}

void ClangdServer::getAST(PathRef File, Range R,
                          Callback<llvm::Optional<ASTNode>> CB) {
  auto Action =
      [R, CB(std::move(CB))](llvm::Expected<InputsAndAST> Inputs) mutable {
        if (!Inputs)
          return CB(Inputs.takeError());
        unsigned Start, End;
        if (auto Offset = positionToOffset(Inputs->Inputs.Contents, R.start))
          Start = *Offset;
        else
          return CB(Offset.takeError());
        if (auto Offset = positionToOffset(Inputs->Inputs.Contents, R.end))
          End = *Offset;
        else
          return CB(Offset.takeError());

        bool Success = SelectionTree::createEach(
            Inputs->AST.getASTContext(), Inputs->AST.getTokens(), Start, End,
            [&](SelectionTree T) {
              if (const SelectionTree::Node *N = T.commonAncestor()) {
                CB(dumpAST(N->ASTNode, Inputs->AST.getTokens(),
                           Inputs->AST.getASTContext()));
                return true;
              }
              return false;
            });
        if (!Success)
          CB(llvm::None);
      };
  WorkScheduler.runWithAST("GetAST", File, std::move(Action));
}

void ClangdServer::customAction(PathRef File, llvm::StringRef Name,
                                Callback<InputsAndAST> Action) {
  WorkScheduler.runWithAST(Name, File, std::move(Action));
}

llvm::StringMap<TUScheduler::FileStats> ClangdServer::fileStats() const {
  return WorkScheduler.fileStats();
}

Context ClangdServer::createProcessingContext(PathRef File) const {
  if (!ConfigProvider)
    return Context::current().clone();

  config::Params Params;
  // Don't reread config files excessively often.
  // FIXME: when we see a config file change event, use the event timestamp.
  Params.FreshTime = std::chrono::steady_clock::now() - std::chrono::seconds(5);
  llvm::SmallString<256> PosixPath;
  if (!File.empty()) {
    assert(llvm::sys::path::is_absolute(File));
    llvm::sys::path::native(File, PosixPath, llvm::sys::path::Style::posix);
    Params.Path = PosixPath.str();
  }

  auto DiagnosticHandler = [](const llvm::SMDiagnostic &Diag) {
    if (Diag.getKind() == llvm::SourceMgr::DK_Error) {
      elog("config error at {0}:{1}:{2}: {3}", Diag.getFilename(),
           Diag.getLineNo(), Diag.getColumnNo(), Diag.getMessage());
    } else {
      log("config warning at {0}:{1}:{2}: {3}", Diag.getFilename(),
          Diag.getLineNo(), Diag.getColumnNo(), Diag.getMessage());
    }
  };
  Config C = ConfigProvider->getConfig(Params, DiagnosticHandler);
  return Context::current().derive(Config::Key, std::move(C));
}

LLVM_NODISCARD bool
ClangdServer::blockUntilIdleForTest(llvm::Optional<double> TimeoutSeconds) {
  return WorkScheduler.blockUntilIdle(timeoutSeconds(TimeoutSeconds)) &&
         (!BackgroundIdx ||
          BackgroundIdx->blockUntilIdleForTest(TimeoutSeconds));
}

void ClangdServer::profile(MemoryTree &MT) const {
  if (DynamicIdx)
    DynamicIdx->profile(MT.child("dynamic_index"));
  if (BackgroundIdx)
    BackgroundIdx->profile(MT.child("background_index"));
  WorkScheduler.profile(MT.child("tuscheduler"));
}
} // namespace clangd
} // namespace clang
