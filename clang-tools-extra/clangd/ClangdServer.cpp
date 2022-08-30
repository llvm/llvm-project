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
#include "Diagnostics.h"
#include "DumpAST.h"
#include "FindSymbols.h"
#include "Format.h"
#include "HeaderSourceSwitch.h"
#include "InlayHints.h"
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
#include "index/StdLib.h"
#include "refactor/Rename.h"
#include "refactor/Tweak.h"
#include "support/Cancellation.h"
#include "support/Logger.h"
#include "support/MemoryTree.h"
#include "support/ThreadsafeFS.h"
#include "support/Trace.h"
#include "clang/Format/Format.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
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
                       const ThreadsafeFS &TFS, AsyncTaskRunner *Tasks)
      : FIndex(FIndex), ServerCallbacks(ServerCallbacks), TFS(TFS),
        Tasks(Tasks) {}

  void onPreambleAST(PathRef Path, llvm::StringRef Version,
                     const CompilerInvocation &CI, ASTContext &Ctx,
                     Preprocessor &PP,
                     const CanonicalIncludes &CanonIncludes) override {
    // If this preamble uses a standard library we haven't seen yet, index it.
    if (FIndex)
      if (auto Loc = Stdlib.add(*CI.getLangOpts(), PP.getHeaderSearchInfo()))
        indexStdlib(CI, std::move(*Loc));

    if (FIndex)
      FIndex->updatePreamble(Path, Version, Ctx, PP, CanonIncludes);
  }

  void indexStdlib(const CompilerInvocation &CI, StdLibLocation Loc) {
    auto Task = [this, LO(*CI.getLangOpts()), Loc(std::move(Loc)),
                 CI(std::make_unique<CompilerInvocation>(CI))]() mutable {
      IndexFileIn IF;
      IF.Symbols = indexStandardLibrary(std::move(CI), Loc, TFS);
      if (Stdlib.isBest(LO))
        FIndex->updatePreamble(std::move(IF));
    };
    if (Tasks)
      // This doesn't have a semaphore to enforce -j, but it's rare.
      Tasks->runAsync("IndexStdlib", std::move(Task));
    else
      Task();
  }

  void onMainAST(PathRef Path, ParsedAST &AST, PublishFn Publish) override {
    if (FIndex)
      FIndex->updateMain(Path, AST);

    assert(AST.getDiagnostics() &&
           "We issue callback only with fresh preambles");
    std::vector<Diag> Diagnostics = *AST.getDiagnostics();
    if (ServerCallbacks)
      Publish([&]() {
        ServerCallbacks->onDiagnosticsReady(Path, AST.version(),
                                            std::move(Diagnostics));
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

  void onPreamblePublished(PathRef File) override {
    if (ServerCallbacks)
      ServerCallbacks->onSemanticsMaybeChanged(File);
  }

private:
  FileIndex *FIndex;
  ClangdServer::Callbacks *ServerCallbacks;
  const ThreadsafeFS &TFS;
  StdLibSet Stdlib;
  AsyncTaskRunner *Tasks;
};

class DraftStoreFS : public ThreadsafeFS {
public:
  DraftStoreFS(const ThreadsafeFS &Base, const DraftStore &Drafts)
      : Base(Base), DirtyFiles(Drafts) {}

private:
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> viewImpl() const override {
    auto OFS = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(
        Base.view(llvm::None));
    OFS->pushOverlay(DirtyFiles.asVFS());
    return OFS;
  }

  const ThreadsafeFS &Base;
  const DraftStore &DirtyFiles;
};

} // namespace

ClangdServer::Options ClangdServer::optsForTest() {
  ClangdServer::Options Opts;
  Opts.UpdateDebounce = DebouncePolicy::fixed(/*zero*/ {});
  Opts.StorePreamblesInMemory = true;
  Opts.AsyncThreadsCount = 4; // Consistent!
  return Opts;
}

ClangdServer::Options::operator TUScheduler::Options() const {
  TUScheduler::Options Opts;
  Opts.AsyncThreadsCount = AsyncThreadsCount;
  Opts.RetentionPolicy = RetentionPolicy;
  Opts.StorePreamblesInMemory = StorePreamblesInMemory;
  Opts.UpdateDebounce = UpdateDebounce;
  Opts.ContextProvider = ContextProvider;
  Opts.PreambleThrottler = PreambleThrottler;
  return Opts;
}

ClangdServer::ClangdServer(const GlobalCompilationDatabase &CDB,
                           const ThreadsafeFS &TFS, const Options &Opts,
                           Callbacks *Callbacks)
    : FeatureModules(Opts.FeatureModules), CDB(CDB), TFS(TFS),
      DynamicIdx(Opts.BuildDynamicSymbolIndex ? new FileIndex() : nullptr),
      ClangTidyProvider(Opts.ClangTidyProvider),
      UseDirtyHeaders(Opts.UseDirtyHeaders),
      LineFoldingOnly(Opts.LineFoldingOnly),
      PreambleParseForwardingFunctions(Opts.PreambleParseForwardingFunctions),
      WorkspaceRoot(Opts.WorkspaceRoot),
      Transient(Opts.ImplicitCancellation ? TUScheduler::InvalidateOnUpdate
                                          : TUScheduler::NoInvalidation),
      DirtyFS(std::make_unique<DraftStoreFS>(TFS, DraftMgr)) {
  if (Opts.AsyncThreadsCount != 0)
    IndexTasks.emplace();
  // Pass a callback into `WorkScheduler` to extract symbols from a newly
  // parsed file and rebuild the file index synchronously each time an AST
  // is parsed.
  WorkScheduler.emplace(CDB, TUScheduler::Options(Opts),
                        std::make_unique<UpdateIndexCallbacks>(
                            DynamicIdx.get(), Callbacks, TFS,
                            IndexTasks ? IndexTasks.getPointer() : nullptr));
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
    BGOpts.ContextProvider = Opts.ContextProvider;
    BackgroundIdx = std::make_unique<BackgroundIndex>(
        TFS, CDB,
        BackgroundIndexStorage::createDiskBackedStorageFactory(
            [&CDB](llvm::StringRef File) { return CDB.getProjectInfo(File); }),
        std::move(BGOpts));
    AddIndex(BackgroundIdx.get());
  }
  if (DynamicIdx)
    AddIndex(DynamicIdx.get());

  if (Opts.FeatureModules) {
    FeatureModule::Facilities F{
        *this->WorkScheduler,
        this->Index,
        this->TFS,
    };
    for (auto &Mod : *Opts.FeatureModules)
      Mod.initialize(F);
  }
}

ClangdServer::~ClangdServer() {
  // Destroying TUScheduler first shuts down request threads that might
  // otherwise access members concurrently.
  // (Nobody can be using TUScheduler because we're on the main thread).
  WorkScheduler.reset();
  // Now requests have stopped, we can shut down feature modules.
  if (FeatureModules) {
    for (auto &Mod : *FeatureModules)
      Mod.stop();
    for (auto &Mod : *FeatureModules)
      Mod.blockUntilIdle(Deadline::infinity());
  }
}

void ClangdServer::addDocument(PathRef File, llvm::StringRef Contents,
                               llvm::StringRef Version,
                               WantDiagnostics WantDiags, bool ForceRebuild) {
  std::string ActualVersion = DraftMgr.addDraft(File, Version, Contents);
  ParseOptions Opts;
  Opts.PreambleParseForwardingFunctions = PreambleParseForwardingFunctions;

  // Compile command is set asynchronously during update, as it can be slow.
  ParseInputs Inputs;
  Inputs.TFS = &getHeaderFS();
  Inputs.Contents = std::string(Contents);
  Inputs.Version = std::move(ActualVersion);
  Inputs.ForceRebuild = ForceRebuild;
  Inputs.Opts = std::move(Opts);
  Inputs.Index = Index;
  Inputs.ClangTidyProvider = ClangTidyProvider;
  Inputs.FeatureModules = FeatureModules;
  bool NewFile = WorkScheduler->update(File, Inputs, WantDiags);
  // If we loaded Foo.h, we want to make sure Foo.cpp is indexed.
  if (NewFile && BackgroundIdx)
    BackgroundIdx->boostRelated(File);
}

void ClangdServer::reparseOpenFilesIfNeeded(
    llvm::function_ref<bool(llvm::StringRef File)> Filter) {
  // Reparse only opened files that were modified.
  for (const Path &FilePath : DraftMgr.getActiveFiles())
    if (Filter(FilePath))
      if (auto Draft = DraftMgr.getDraft(FilePath)) // else disappeared in race?
        addDocument(FilePath, *Draft->Contents, Draft->Version,
                    WantDiagnostics::Auto);
}

std::shared_ptr<const std::string> ClangdServer::getDraft(PathRef File) const {
  auto Draft = DraftMgr.getDraft(File);
  if (!Draft)
    return nullptr;
  return std::move(Draft->Contents);
}

std::function<Context(PathRef)>
ClangdServer::createConfiguredContextProvider(const config::Provider *Provider,
                                              Callbacks *Publish) {
  if (!Provider)
    return [](llvm::StringRef) { return Context::current().clone(); };

  struct Impl {
    const config::Provider *Provider;
    ClangdServer::Callbacks *Publish;
    std::mutex PublishMu;

    Impl(const config::Provider *Provider, ClangdServer::Callbacks *Publish)
        : Provider(Provider), Publish(Publish) {}

    Context operator()(llvm::StringRef File) {
      config::Params Params;
      // Don't reread config files excessively often.
      // FIXME: when we see a config file change event, use the event timestamp?
      Params.FreshTime =
          std::chrono::steady_clock::now() - std::chrono::seconds(5);
      llvm::SmallString<256> PosixPath;
      if (!File.empty()) {
        assert(llvm::sys::path::is_absolute(File));
        llvm::sys::path::native(File, PosixPath, llvm::sys::path::Style::posix);
        Params.Path = PosixPath.str();
      }

      llvm::StringMap<std::vector<Diag>> ReportableDiagnostics;
      Config C = Provider->getConfig(Params, [&](const llvm::SMDiagnostic &D) {
        // Create the map entry even for note diagnostics we don't report.
        // This means that when the file is parsed with no warnings, we
        // publish an empty set of diagnostics, clearing any the client has.
        handleDiagnostic(D, !Publish || D.getFilename().empty()
                                ? nullptr
                                : &ReportableDiagnostics[D.getFilename()]);
      });
      // Blindly publish diagnostics for the (unopened) parsed config files.
      // We must avoid reporting diagnostics for *the same file* concurrently.
      // Source diags are published elsewhere, but those are different files.
      if (!ReportableDiagnostics.empty()) {
        std::lock_guard<std::mutex> Lock(PublishMu);
        for (auto &Entry : ReportableDiagnostics)
          Publish->onDiagnosticsReady(Entry.first(), /*Version=*/"",
                                      std::move(Entry.second));
      }
      return Context::current().derive(Config::Key, std::move(C));
    }

    void handleDiagnostic(const llvm::SMDiagnostic &D,
                          std::vector<Diag> *ClientDiagnostics) {
      switch (D.getKind()) {
      case llvm::SourceMgr::DK_Error:
        elog("config error at {0}:{1}:{2}: {3}", D.getFilename(), D.getLineNo(),
             D.getColumnNo(), D.getMessage());
        break;
      case llvm::SourceMgr::DK_Warning:
        log("config warning at {0}:{1}:{2}: {3}", D.getFilename(),
            D.getLineNo(), D.getColumnNo(), D.getMessage());
        break;
      case llvm::SourceMgr::DK_Note:
      case llvm::SourceMgr::DK_Remark:
        vlog("config note at {0}:{1}:{2}: {3}", D.getFilename(), D.getLineNo(),
             D.getColumnNo(), D.getMessage());
        ClientDiagnostics = nullptr; // Don't emit notes as LSP diagnostics.
        break;
      }
      if (ClientDiagnostics)
        ClientDiagnostics->push_back(toDiag(D, Diag::ClangdConfig));
    }
  };

  // Copyable wrapper.
  return [I(std::make_shared<Impl>(Provider, Publish))](llvm::StringRef Path) {
    return (*I)(Path);
  };
}

void ClangdServer::removeDocument(PathRef File) {
  DraftMgr.removeDraft(File);
  WorkScheduler->remove(File);
}

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
    } else if (CodeCompleteOpts.Index) {
      SpecFuzzyFind.emplace();
      {
        std::lock_guard<std::mutex> Lock(CachedCompletionFuzzyFindRequestMutex);
        SpecFuzzyFind->CachedReq = CachedCompletionFuzzyFindRequestByFile[File];
      }
    }
    ParseInputs ParseInput{IP->Command, &getHeaderFS(), IP->Contents.str()};
    // FIXME: Add traling new line if there is none at eof, workaround a crash,
    // see https://github.com/clangd/clangd/issues/332
    if (!IP->Contents.endswith("\n"))
      ParseInput.Contents.append("\n");
    ParseInput.Index = Index;

    CodeCompleteOpts.MainFileSignals = IP->Signals;
    CodeCompleteOpts.AllScopes = Config::current().Completion.AllScopes;
    // FIXME(ibiryukov): even if Preamble is non-null, we may want to check
    // both the old and the new version in case only one of them matches.
    CodeCompleteResult Result = clangd::codeComplete(
        File, Pos, IP->Preamble, ParseInput, CodeCompleteOpts,
        SpecFuzzyFind ? SpecFuzzyFind.getPointer() : nullptr);
    {
      clang::clangd::trace::Span Tracer("Completion results callback");
      CB(std::move(Result));
    }
    if (SpecFuzzyFind && SpecFuzzyFind->NewReq) {
      std::lock_guard<std::mutex> Lock(CachedCompletionFuzzyFindRequestMutex);
      CachedCompletionFuzzyFindRequestByFile[File] =
          SpecFuzzyFind->NewReq.value();
    }
    // SpecFuzzyFind is only destroyed after speculative fuzzy find finishes.
    // We don't want `codeComplete` to wait for the async call if it doesn't use
    // the result (e.g. non-index completion, speculation fails), so that `CB`
    // is called as soon as results are available.
  };

  // We use a potentially-stale preamble because latency is critical here.
  WorkScheduler->runWithPreamble(
      "CodeComplete", File,
      (Opts.RunParser == CodeCompleteOptions::AlwaysParse)
          ? TUScheduler::Stale
          : TUScheduler::StaleOrAbsent,
      std::move(Task));
}

void ClangdServer::signatureHelp(PathRef File, Position Pos,
                                 MarkupKind DocumentationFormat,
                                 Callback<SignatureHelp> CB) {

  auto Action = [Pos, File = File.str(), CB = std::move(CB),
                 DocumentationFormat,
                 this](llvm::Expected<InputsAndPreamble> IP) mutable {
    if (!IP)
      return CB(IP.takeError());

    const auto *PreambleData = IP->Preamble;
    if (!PreambleData)
      return CB(error("Failed to parse includes"));

    ParseInputs ParseInput{IP->Command, &getHeaderFS(), IP->Contents.str()};
    // FIXME: Add traling new line if there is none at eof, workaround a crash,
    // see https://github.com/clangd/clangd/issues/332
    if (!IP->Contents.endswith("\n"))
      ParseInput.Contents.append("\n");
    ParseInput.Index = Index;
    CB(clangd::signatureHelp(File, Pos, *PreambleData, ParseInput,
                             DocumentationFormat));
  };

  // Unlike code completion, we wait for a preamble here.
  WorkScheduler->runWithPreamble("SignatureHelp", File, TUScheduler::Stale,
                                 std::move(Action));
}

void ClangdServer::formatFile(PathRef File, llvm::Optional<Range> Rng,
                              Callback<tooling::Replacements> CB) {
  auto Code = getDraft(File);
  if (!Code)
    return CB(llvm::make_error<LSPError>("trying to format non-added document",
                                         ErrorCode::InvalidParams));
  tooling::Range RequestedRange;
  if (Rng) {
    llvm::Expected<size_t> Begin = positionToOffset(*Code, Rng->start);
    if (!Begin)
      return CB(Begin.takeError());
    llvm::Expected<size_t> End = positionToOffset(*Code, Rng->end);
    if (!End)
      return CB(End.takeError());
    RequestedRange = tooling::Range(*Begin, *End - *Begin);
  } else {
    RequestedRange = tooling::Range(0, Code->size());
  }

  // Call clang-format.
  auto Action = [File = File.str(), Code = std::move(*Code),
                 Ranges = std::vector<tooling::Range>{RequestedRange},
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
  WorkScheduler->runQuick("Format", File, std::move(Action));
}

void ClangdServer::formatOnType(PathRef File, Position Pos,
                                StringRef TriggerText,
                                Callback<std::vector<TextEdit>> CB) {
  auto Code = getDraft(File);
  if (!Code)
    return CB(llvm::make_error<LSPError>("trying to format non-added document",
                                         ErrorCode::InvalidParams));
  llvm::Expected<size_t> CursorPos = positionToOffset(*Code, Pos);
  if (!CursorPos)
    return CB(CursorPos.takeError());
  auto Action = [File = File.str(), Code = std::move(*Code),
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
  WorkScheduler->runQuick("FormatOnType", File, std::move(Action));
}

void ClangdServer::prepareRename(PathRef File, Position Pos,
                                 llvm::Optional<std::string> NewName,
                                 const RenameOptions &RenameOpts,
                                 Callback<RenameResult> CB) {
  auto Action = [Pos, File = File.str(), CB = std::move(CB),
                 NewName = std::move(NewName),
                 RenameOpts](llvm::Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    // prepareRename is latency-sensitive: we don't query the index, as we
    // only need main-file references
    auto Results =
        clangd::rename({Pos, NewName.value_or("__clangd_rename_placeholder"),
                        InpAST->AST, File, /*FS=*/nullptr,
                        /*Index=*/nullptr, RenameOpts});
    if (!Results) {
      // LSP says to return null on failure, but that will result in a generic
      // failure message. If we send an LSP error response, clients can surface
      // the message to users (VSCode does).
      return CB(Results.takeError());
    }
    return CB(*Results);
  };
  WorkScheduler->runWithAST("PrepareRename", File, std::move(Action));
}

void ClangdServer::rename(PathRef File, Position Pos, llvm::StringRef NewName,
                          const RenameOptions &Opts,
                          Callback<RenameResult> CB) {
  auto Action = [File = File.str(), NewName = NewName.str(), Pos, Opts,
                 CB = std::move(CB),
                 this](llvm::Expected<InputsAndAST> InpAST) mutable {
    // Tracks number of files edited per invocation.
    static constexpr trace::Metric RenameFiles("rename_files",
                                               trace::Metric::Distribution);
    if (!InpAST)
      return CB(InpAST.takeError());
    auto R = clangd::rename({Pos, NewName, InpAST->AST, File,
                             DirtyFS->view(llvm::None), Index, Opts});
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
  WorkScheduler->runWithAST("Rename", File, std::move(Action));
}

// May generate several candidate selections, due to SelectionTree ambiguity.
// vector of pointers because GCC doesn't like non-copyable Selection.
static llvm::Expected<std::vector<std::unique_ptr<Tweak::Selection>>>
tweakSelection(const Range &Sel, const InputsAndAST &AST,
               llvm::vfs::FileSystem *FS) {
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
            AST.Inputs.Index, AST.AST, *Begin, *End, std::move(T), FS));
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
  auto Action = [Sel, CB = std::move(CB), Filter = std::move(Filter),
                 FeatureModules(this->FeatureModules)](
                    Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    auto Selections = tweakSelection(Sel, *InpAST, /*FS=*/nullptr);
    if (!Selections)
      return CB(Selections.takeError());
    std::vector<TweakRef> Res;
    // Don't allow a tweak to fire more than once across ambiguous selections.
    llvm::DenseSet<llvm::StringRef> PreparedTweaks;
    auto DeduplicatingFilter = [&](const Tweak &T) {
      return Filter(T) && !PreparedTweaks.count(T.id());
    };
    for (const auto &Sel : *Selections) {
      for (auto &T : prepareTweaks(*Sel, DeduplicatingFilter, FeatureModules)) {
        Res.push_back({T->id(), T->title(), T->kind()});
        PreparedTweaks.insert(T->id());
        TweakAvailable.record(1, T->id());
      }
    }

    CB(std::move(Res));
  };

  WorkScheduler->runWithAST("EnumerateTweaks", File, std::move(Action),
                            Transient);
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
    auto FS = DirtyFS->view(llvm::None);
    auto Selections = tweakSelection(Sel, *InpAST, FS.get());
    if (!Selections)
      return CB(Selections.takeError());
    llvm::Optional<llvm::Expected<Tweak::Effect>> Effect;
    // Try each selection, take the first one that prepare()s.
    // If they all fail, Effect will hold get the last error.
    for (const auto &Selection : *Selections) {
      auto T = prepareTweak(TweakID, *Selection, FeatureModules);
      if (T) {
        Effect = (*T)->apply(*Selection);
        break;
      }
      Effect = T.takeError();
    }
    assert(Effect && "Expected at least one selection");
    if (*Effect && (*Effect)->FormatEdits) {
      // Format tweaks that require it centrally here.
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
  WorkScheduler->runWithAST("ApplyTweak", File, std::move(Action));
}

void ClangdServer::locateSymbolAt(PathRef File, Position Pos,
                                  Callback<std::vector<LocatedSymbol>> CB) {
  auto Action = [Pos, CB = std::move(CB),
                 this](llvm::Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::locateSymbolAt(InpAST->AST, Pos, Index));
  };

  WorkScheduler->runWithAST("Definitions", File, std::move(Action));
}

void ClangdServer::switchSourceHeader(
    PathRef Path, Callback<llvm::Optional<clangd::Path>> CB) {
  // We want to return the result as fast as possible, strategy is:
  //  1) use the file-only heuristic, it requires some IO but it is much
  //     faster than building AST, but it only works when .h/.cc files are in
  //     the same directory.
  //  2) if 1) fails, we use the AST&Index approach, it is slower but supports
  //     different code layout.
  if (auto CorrespondingFile =
          getCorrespondingHeaderOrSource(Path, TFS.view(llvm::None)))
    return CB(std::move(CorrespondingFile));
  auto Action = [Path = Path.str(), CB = std::move(CB),
                 this](llvm::Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(getCorrespondingHeaderOrSource(Path, InpAST->AST, Index));
  };
  WorkScheduler->runWithAST("SwitchHeaderSource", Path, std::move(Action));
}

void ClangdServer::findDocumentHighlights(
    PathRef File, Position Pos, Callback<std::vector<DocumentHighlight>> CB) {
  auto Action =
      [Pos, CB = std::move(CB)](llvm::Expected<InputsAndAST> InpAST) mutable {
        if (!InpAST)
          return CB(InpAST.takeError());
        CB(clangd::findDocumentHighlights(InpAST->AST, Pos));
      };

  WorkScheduler->runWithAST("Highlights", File, std::move(Action), Transient);
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

  WorkScheduler->runWithAST("Hover", File, std::move(Action), Transient);
}

void ClangdServer::typeHierarchy(PathRef File, Position Pos, int Resolve,
                                 TypeHierarchyDirection Direction,
                                 Callback<std::vector<TypeHierarchyItem>> CB) {
  auto Action = [File = File.str(), Pos, Resolve, Direction, CB = std::move(CB),
                 this](Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::getTypeHierarchy(InpAST->AST, Pos, Resolve, Direction, Index,
                                File));
  };

  WorkScheduler->runWithAST("TypeHierarchy", File, std::move(Action));
}

void ClangdServer::superTypes(
    const TypeHierarchyItem &Item,
    Callback<llvm::Optional<std::vector<TypeHierarchyItem>>> CB) {
  WorkScheduler->run("typeHierarchy/superTypes", /*Path=*/"",
                     [=, CB = std::move(CB)]() mutable {
                       CB(clangd::superTypes(Item, Index));
                     });
}

void ClangdServer::subTypes(const TypeHierarchyItem &Item,
                            Callback<std::vector<TypeHierarchyItem>> CB) {
  WorkScheduler->run(
      "typeHierarchy/subTypes", /*Path=*/"",
      [=, CB = std::move(CB)]() mutable { CB(clangd::subTypes(Item, Index)); });
}

void ClangdServer::resolveTypeHierarchy(
    TypeHierarchyItem Item, int Resolve, TypeHierarchyDirection Direction,
    Callback<llvm::Optional<TypeHierarchyItem>> CB) {
  WorkScheduler->run(
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
  WorkScheduler->runWithAST("CallHierarchy", File, std::move(Action));
}

void ClangdServer::incomingCalls(
    const CallHierarchyItem &Item,
    Callback<std::vector<CallHierarchyIncomingCall>> CB) {
  WorkScheduler->run("Incoming Calls", "",
                     [CB = std::move(CB), Item, this]() mutable {
                       CB(clangd::incomingCalls(Item, Index));
                     });
}

void ClangdServer::inlayHints(PathRef File, llvm::Optional<Range> RestrictRange,
                              Callback<std::vector<InlayHint>> CB) {
  auto Action = [RestrictRange(std::move(RestrictRange)),
                 CB = std::move(CB)](Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::inlayHints(InpAST->AST, std::move(RestrictRange)));
  };
  WorkScheduler->runWithAST("InlayHints", File, std::move(Action), Transient);
}

void ClangdServer::onFileEvent(const DidChangeWatchedFilesParams &Params) {
  // FIXME: Do nothing for now. This will be used for indexing and potentially
  // invalidating other caches.
}

void ClangdServer::workspaceSymbols(
    llvm::StringRef Query, int Limit,
    Callback<std::vector<SymbolInformation>> CB) {
  WorkScheduler->run(
      "getWorkspaceSymbols", /*Path=*/"",
      [Query = Query.str(), Limit, CB = std::move(CB), this]() mutable {
        CB(clangd::getWorkspaceSymbols(Query, Limit, Index,
                                       WorkspaceRoot.value_or("")));
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
  WorkScheduler->runWithAST("DocumentSymbols", File, std::move(Action),
                            Transient);
}

void ClangdServer::foldingRanges(llvm::StringRef File,
                                 Callback<std::vector<FoldingRange>> CB) {
  auto Code = getDraft(File);
  if (!Code)
    return CB(llvm::make_error<LSPError>(
        "trying to compute folding ranges for non-added document",
        ErrorCode::InvalidParams));
  auto Action = [LineFoldingOnly = LineFoldingOnly, CB = std::move(CB),
                 Code = std::move(*Code)]() mutable {
    CB(clangd::getFoldingRanges(Code, LineFoldingOnly));
  };
  // We want to make sure folding ranges are always available for all the open
  // files, hence prefer runQuick to not wait for operations on other files.
  WorkScheduler->runQuick("FoldingRanges", File, std::move(Action));
}

void ClangdServer::findType(llvm::StringRef File, Position Pos,
                            Callback<std::vector<LocatedSymbol>> CB) {
  auto Action =
      [Pos, CB = std::move(CB)](llvm::Expected<InputsAndAST> InpAST) mutable {
        if (!InpAST)
          return CB(InpAST.takeError());
        CB(clangd::findType(InpAST->AST, Pos));
      };
  WorkScheduler->runWithAST("FindType", File, std::move(Action));
}

void ClangdServer::findImplementations(
    PathRef File, Position Pos, Callback<std::vector<LocatedSymbol>> CB) {
  auto Action = [Pos, CB = std::move(CB),
                 this](llvm::Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::findImplementations(InpAST->AST, Pos, Index));
  };

  WorkScheduler->runWithAST("Implementations", File, std::move(Action));
}

void ClangdServer::findReferences(PathRef File, Position Pos, uint32_t Limit,
                                  Callback<ReferencesResult> CB) {
  auto Action = [Pos, Limit, CB = std::move(CB),
                 this](llvm::Expected<InputsAndAST> InpAST) mutable {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::findReferences(InpAST->AST, Pos, Limit, Index));
  };

  WorkScheduler->runWithAST("References", File, std::move(Action));
}

void ClangdServer::symbolInfo(PathRef File, Position Pos,
                              Callback<std::vector<SymbolDetails>> CB) {
  auto Action =
      [Pos, CB = std::move(CB)](llvm::Expected<InputsAndAST> InpAST) mutable {
        if (!InpAST)
          return CB(InpAST.takeError());
        CB(clangd::getSymbolInfo(InpAST->AST, Pos));
      };

  WorkScheduler->runWithAST("SymbolInfo", File, std::move(Action));
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
  WorkScheduler->runWithAST("SemanticRanges", File, std::move(Action));
}

void ClangdServer::documentLinks(PathRef File,
                                 Callback<std::vector<DocumentLink>> CB) {
  auto Action =
      [CB = std::move(CB)](llvm::Expected<InputsAndAST> InpAST) mutable {
        if (!InpAST)
          return CB(InpAST.takeError());
        CB(clangd::getDocumentLinks(InpAST->AST));
      };
  WorkScheduler->runWithAST("DocumentLinks", File, std::move(Action),
                            Transient);
}

void ClangdServer::semanticHighlights(
    PathRef File, Callback<std::vector<HighlightingToken>> CB) {
  auto Action =
      [CB = std::move(CB)](llvm::Expected<InputsAndAST> InpAST) mutable {
        if (!InpAST)
          return CB(InpAST.takeError());
        CB(clangd::getSemanticHighlightings(InpAST->AST));
      };
  WorkScheduler->runWithAST("SemanticHighlights", File, std::move(Action),
                            Transient);
}

void ClangdServer::getAST(PathRef File, llvm::Optional<Range> R,
                          Callback<llvm::Optional<ASTNode>> CB) {
  auto Action =
      [R, CB(std::move(CB))](llvm::Expected<InputsAndAST> Inputs) mutable {
        if (!Inputs)
          return CB(Inputs.takeError());
        if (!R) {
          // It's safe to pass in the TU, as dumpAST() does not
          // deserialize the preamble.
          auto Node = DynTypedNode::create(
              *Inputs->AST.getASTContext().getTranslationUnitDecl());
          return CB(dumpAST(Node, Inputs->AST.getTokens(),
                            Inputs->AST.getASTContext()));
        }
        unsigned Start, End;
        if (auto Offset = positionToOffset(Inputs->Inputs.Contents, R->start))
          Start = *Offset;
        else
          return CB(Offset.takeError());
        if (auto Offset = positionToOffset(Inputs->Inputs.Contents, R->end))
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
  WorkScheduler->runWithAST("GetAST", File, std::move(Action));
}

void ClangdServer::customAction(PathRef File, llvm::StringRef Name,
                                Callback<InputsAndAST> Action) {
  WorkScheduler->runWithAST(Name, File, std::move(Action));
}

void ClangdServer::diagnostics(PathRef File, Callback<std::vector<Diag>> CB) {
  auto Action =
      [CB = std::move(CB)](llvm::Expected<InputsAndAST> InpAST) mutable {
        if (!InpAST)
          return CB(InpAST.takeError());
        if (auto Diags = InpAST->AST.getDiagnostics())
          return CB(*Diags);
        // FIXME: Use ServerCancelled error once it is settled in LSP-3.17.
        return CB(llvm::make_error<LSPError>("server is busy parsing includes",
                                             ErrorCode::InternalError));
      };

  WorkScheduler->runWithAST("Diagnostics", File, std::move(Action));
}

llvm::StringMap<TUScheduler::FileStats> ClangdServer::fileStats() const {
  return WorkScheduler->fileStats();
}

[[nodiscard]] bool
ClangdServer::blockUntilIdleForTest(llvm::Optional<double> TimeoutSeconds) {
  // Order is important here: we don't want to block on A and then B,
  // if B might schedule work on A.

  // Nothing else can schedule work on TUScheduler, because it's not threadsafe
  // and we're blocking the main thread.
  if (!WorkScheduler->blockUntilIdle(timeoutSeconds(TimeoutSeconds)))
    return false;
  // TUScheduler is the only thing that starts background indexing work.
  if (IndexTasks && !IndexTasks->wait(timeoutSeconds(TimeoutSeconds)))
    return false;

  // Unfortunately we don't have strict topological order between the rest of
  // the components. E.g. CDB broadcast triggers backrgound indexing.
  // This queries the CDB which may discover new work if disk has changed.
  //
  // So try each one a few times in a loop.
  // If there are no tricky interactions then all after the first are no-ops.
  // Then on the last iteration, verify they're idle without waiting.
  //
  // There's a small chance they're juggling work and we didn't catch them :-(
  for (llvm::Optional<double> Timeout :
       {TimeoutSeconds, TimeoutSeconds, llvm::Optional<double>(0)}) {
    if (!CDB.blockUntilIdle(timeoutSeconds(Timeout)))
      return false;
    if (BackgroundIdx && !BackgroundIdx->blockUntilIdleForTest(Timeout))
      return false;
    if (FeatureModules && llvm::any_of(*FeatureModules, [&](FeatureModule &M) {
          return !M.blockUntilIdle(timeoutSeconds(Timeout));
        }))
      return false;
  }

  assert(WorkScheduler->blockUntilIdle(Deadline::zero()) &&
         "Something scheduled work while we're blocking the main thread!");
  return true;
}

void ClangdServer::profile(MemoryTree &MT) const {
  if (DynamicIdx)
    DynamicIdx->profile(MT.child("dynamic_index"));
  if (BackgroundIdx)
    BackgroundIdx->profile(MT.child("background_index"));
  WorkScheduler->profile(MT.child("tuscheduler"));
}
} // namespace clangd
} // namespace clang
