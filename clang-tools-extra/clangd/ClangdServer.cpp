//===--- ClangdServer.cpp - Main clangd server code --------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------===//

#include "ClangdServer.h"
#include "ClangdUnit.h"
#include "CodeComplete.h"
#include "FindSymbols.h"
#include "Headers.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "TUScheduler.h"
#include "Trace.h"
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
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <future>
#include <memory>
#include <mutex>

namespace clang {
namespace clangd {
namespace {

// Update the FileIndex with new ASTs and plumb the diagnostics responses.
struct UpdateIndexCallbacks : public ParsingCallbacks {
  UpdateIndexCallbacks(FileIndex *FIndex, DiagnosticsConsumer &DiagConsumer)
      : FIndex(FIndex), DiagConsumer(DiagConsumer) {}

  void onPreambleAST(PathRef Path, ASTContext &Ctx,
                     std::shared_ptr<clang::Preprocessor> PP,
                     const CanonicalIncludes &CanonIncludes) override {
    if (FIndex)
      FIndex->updatePreamble(Path, Ctx, std::move(PP), CanonIncludes);
  }

  void onMainAST(PathRef Path, ParsedAST &AST) override {
    if (FIndex)
      FIndex->updateMain(Path, AST);
  }

  void onDiagnostics(PathRef File, std::vector<Diag> Diags) override {
    DiagConsumer.onDiagnosticsReady(File, std::move(Diags));
  }

  void onFileUpdated(PathRef File, const TUStatus &Status) override {
    DiagConsumer.onFileUpdated(File, Status);
  }

private:
  FileIndex *FIndex;
  DiagnosticsConsumer &DiagConsumer;
};
} // namespace

ClangdServer::Options ClangdServer::optsForTest() {
  ClangdServer::Options Opts;
  Opts.UpdateDebounce = std::chrono::steady_clock::duration::zero(); // Faster!
  Opts.StorePreamblesInMemory = true;
  Opts.AsyncThreadsCount = 4; // Consistent!
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
      ClangTidyOptProvider(Opts.ClangTidyOptProvider),
      SuggestMissingIncludes(Opts.SuggestMissingIncludes),
      WorkspaceRoot(Opts.WorkspaceRoot),
      // Pass a callback into `WorkScheduler` to extract symbols from a newly
      // parsed file and rebuild the file index synchronously each time an AST
      // is parsed.
      // FIXME(ioeric): this can be slow and we may be able to index on less
      // critical paths.
      WorkScheduler(CDB, Opts.AsyncThreadsCount, Opts.StorePreamblesInMemory,
                    llvm::make_unique<UpdateIndexCallbacks>(DynamicIdx.get(),
                                                            DiagConsumer),
                    Opts.UpdateDebounce, Opts.RetentionPolicy) {
  // Adds an index to the stack, at higher priority than existing indexes.
  auto AddIndex = [&](SymbolIndex *Idx) {
    if (this->Index != nullptr) {
      MergedIdx.push_back(llvm::make_unique<MergedIndex>(Idx, this->Index));
      this->Index = MergedIdx.back().get();
    } else {
      this->Index = Idx;
    }
  };
  if (Opts.StaticIndex)
    AddIndex(Opts.StaticIndex);
  if (Opts.BackgroundIndex) {
    BackgroundIdx = llvm::make_unique<BackgroundIndex>(
        Context::current().clone(), FSProvider, CDB,
        BackgroundIndexStorage::createDiskBackedStorageFactory(),
        Opts.BackgroundIndexRebuildPeriodMs);
    AddIndex(BackgroundIdx.get());
  }
  if (DynamicIdx)
    AddIndex(DynamicIdx.get());
}

void ClangdServer::addDocument(PathRef File, llvm::StringRef Contents,
                               WantDiagnostics WantDiags) {
  ParseOptions Opts;
  Opts.ClangTidyOpts = tidy::ClangTidyOptions::getDefaults();
  if (ClangTidyOptProvider)
    Opts.ClangTidyOpts = ClangTidyOptProvider->getOptions(File);
  Opts.SuggestMissingIncludes = SuggestMissingIncludes;

  // Compile command is set asynchronously during update, as it can be slow.
  ParseInputs Inputs;
  Inputs.FS = FSProvider.getFileSystem();
  Inputs.Contents = Contents;
  Inputs.Opts = std::move(Opts);
  Inputs.Index = Index;
  WorkScheduler.update(File, Inputs, WantDiags);
}

void ClangdServer::removeDocument(PathRef File) { WorkScheduler.remove(File); }

void ClangdServer::codeComplete(PathRef File, Position Pos,
                                const clangd::CodeCompleteOptions &Opts,
                                Callback<CodeCompleteResult> CB) {
  // Copy completion options for passing them to async task handler.
  auto CodeCompleteOpts = Opts;
  if (!CodeCompleteOpts.Index) // Respect overridden index.
    CodeCompleteOpts.Index = Index;

  auto FS = FSProvider.getFileSystem();
  auto Task = [Pos, FS, CodeCompleteOpts,
               this](Path File, Callback<CodeCompleteResult> CB,
                     llvm::Expected<InputsAndPreamble> IP) {
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
  WorkScheduler.runWithPreamble("CodeComplete", File,
                                Opts.AllowFallback ? TUScheduler::StaleOrAbsent
                                                   : TUScheduler::Stale,
                                Bind(Task, File.str(), std::move(CB)));
}

void ClangdServer::signatureHelp(PathRef File, Position Pos,
                                 Callback<SignatureHelp> CB) {

  auto FS = FSProvider.getFileSystem();
  auto *Index = this->Index;
  auto Action = [Pos, FS, Index](Path File, Callback<SignatureHelp> CB,
                                 llvm::Expected<InputsAndPreamble> IP) {
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
                                Bind(Action, File.str(), std::move(CB)));
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

llvm::Expected<tooling::Replacements>
ClangdServer::formatOnType(llvm::StringRef Code, PathRef File, Position Pos) {
  // Look for the previous opening brace from the character position and
  // format starting from there.
  llvm::Expected<size_t> CursorPos = positionToOffset(Code, Pos);
  if (!CursorPos)
    return CursorPos.takeError();
  size_t PreviousLBracePos =
      llvm::StringRef(Code).find_last_of('{', *CursorPos);
  if (PreviousLBracePos == llvm::StringRef::npos)
    PreviousLBracePos = *CursorPos;
  size_t Len = *CursorPos - PreviousLBracePos;

  return formatCode(Code, File, {tooling::Range(PreviousLBracePos, Len)});
}

void ClangdServer::rename(PathRef File, Position Pos, llvm::StringRef NewName,
                          Callback<std::vector<TextEdit>> CB) {
  auto Action = [Pos](Path File, std::string NewName,
                      Callback<std::vector<TextEdit>> CB,
                      llvm::Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return CB(InpAST.takeError());
    auto Changes = renameWithinFile(InpAST->AST, File, Pos, NewName);
    if (!Changes)
      return CB(Changes.takeError());
    std::vector<TextEdit> Edits;
    for (const auto &Rep : *Changes)
      Edits.push_back(replacementToEdit(InpAST->Inputs.Contents, Rep));
    return CB(std::move(Edits));
  };

  WorkScheduler.runWithAST(
      "Rename", File, Bind(Action, File.str(), NewName.str(), std::move(CB)));
}

static llvm::Expected<Tweak::Selection>
tweakSelection(const Range &Sel, const InputsAndAST &AST) {
  auto Begin = positionToOffset(AST.Inputs.Contents, Sel.start);
  if (!Begin)
    return Begin.takeError();
  auto End = positionToOffset(AST.Inputs.Contents, Sel.end);
  if (!End)
    return End.takeError();
  return Tweak::Selection(AST.AST, *Begin, *End);
}

void ClangdServer::enumerateTweaks(PathRef File, Range Sel,
                                   Callback<std::vector<TweakRef>> CB) {
  auto Action = [Sel](decltype(CB) CB, std::string File,
                      Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return CB(InpAST.takeError());
    auto Selection = tweakSelection(Sel, *InpAST);
    if (!Selection)
      return CB(Selection.takeError());
    std::vector<TweakRef> Res;
    for (auto &T : prepareTweaks(*Selection))
      Res.push_back({T->id(), T->title()});
    CB(std::move(Res));
  };

  WorkScheduler.runWithAST("EnumerateTweaks", File,
                           Bind(Action, std::move(CB), File.str()));
}

void ClangdServer::applyTweak(PathRef File, Range Sel, StringRef TweakID,
                              Callback<tooling::Replacements> CB) {
  auto Action = [Sel](decltype(CB) CB, std::string File, std::string TweakID,
                      Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return CB(InpAST.takeError());
    auto Selection = tweakSelection(Sel, *InpAST);
    if (!Selection)
      return CB(Selection.takeError());
    auto A = prepareTweak(TweakID, *Selection);
    if (!A)
      return CB(A.takeError());
    auto RawReplacements = (*A)->apply(*Selection);
    if (!RawReplacements)
      return CB(RawReplacements.takeError());
    // FIXME: this function has I/O operations (find .clang-format file), figure
    // out a way to cache the format style.
    auto Style = getFormatStyleForFile(File, InpAST->Inputs.Contents,
                                       InpAST->Inputs.FS.get());
    return CB(
        cleanupAndFormat(InpAST->Inputs.Contents, *RawReplacements, Style));
  };
  WorkScheduler.runWithAST(
      "ApplyTweak", File,
      Bind(Action, std::move(CB), File.str(), TweakID.str()));
}

void ClangdServer::dumpAST(PathRef File,
                           llvm::unique_function<void(std::string)> Callback) {
  auto Action = [](decltype(Callback) Callback,
                   llvm::Expected<InputsAndAST> InpAST) {
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

  WorkScheduler.runWithAST("DumpAST", File, Bind(Action, std::move(Callback)));
}

void ClangdServer::locateSymbolAt(PathRef File, Position Pos,
                                  Callback<std::vector<LocatedSymbol>> CB) {
  auto Action = [Pos, this](decltype(CB) CB,
                            llvm::Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::locateSymbolAt(InpAST->AST, Pos, Index));
  };

  WorkScheduler.runWithAST("Definitions", File, Bind(Action, std::move(CB)));
}

llvm::Optional<Path> ClangdServer::switchSourceHeader(PathRef Path) {

  llvm::StringRef SourceExtensions[] = {".cpp", ".c", ".cc", ".cxx",
                                        ".c++", ".m", ".mm"};
  llvm::StringRef HeaderExtensions[] = {".h", ".hh", ".hpp", ".hxx", ".inc"};

  llvm::StringRef PathExt = llvm::sys::path::extension(Path);

  // Lookup in a list of known extensions.
  auto SourceIter =
      llvm::find_if(SourceExtensions, [&PathExt](PathRef SourceExt) {
        return SourceExt.equals_lower(PathExt);
      });
  bool IsSource = SourceIter != std::end(SourceExtensions);

  auto HeaderIter =
      llvm::find_if(HeaderExtensions, [&PathExt](PathRef HeaderExt) {
        return HeaderExt.equals_lower(PathExt);
      });

  bool IsHeader = HeaderIter != std::end(HeaderExtensions);

  // We can only switch between the known extensions.
  if (!IsSource && !IsHeader)
    return None;

  // Array to lookup extensions for the switch. An opposite of where original
  // extension was found.
  llvm::ArrayRef<llvm::StringRef> NewExts;
  if (IsSource)
    NewExts = HeaderExtensions;
  else
    NewExts = SourceExtensions;

  // Storage for the new path.
  llvm::SmallString<128> NewPath = llvm::StringRef(Path);

  // Instance of vfs::FileSystem, used for file existence checks.
  auto FS = FSProvider.getFileSystem();

  // Loop through switched extension candidates.
  for (llvm::StringRef NewExt : NewExts) {
    llvm::sys::path::replace_extension(NewPath, NewExt);
    if (FS->exists(NewPath))
      return NewPath.str().str(); // First str() to convert from SmallString to
                                  // StringRef, second to convert from StringRef
                                  // to std::string

    // Also check NewExt in upper-case, just in case.
    llvm::sys::path::replace_extension(NewPath, NewExt.upper());
    if (FS->exists(NewPath))
      return NewPath.str().str();
  }

  return None;
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
  auto Action = [Pos](Callback<std::vector<DocumentHighlight>> CB,
                      llvm::Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::findDocumentHighlights(InpAST->AST, Pos));
  };

  WorkScheduler.runWithAST("Highlights", File, Bind(Action, std::move(CB)));
}

void ClangdServer::findHover(PathRef File, Position Pos,
                             Callback<llvm::Optional<Hover>> CB) {
  auto Action = [Pos](Callback<llvm::Optional<Hover>> CB,
                      llvm::Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::getHover(InpAST->AST, Pos));
  };

  WorkScheduler.runWithAST("Hover", File, Bind(Action, std::move(CB)));
}

void ClangdServer::typeHierarchy(PathRef File, Position Pos, int Resolve,
                                 TypeHierarchyDirection Direction,
                                 Callback<Optional<TypeHierarchyItem>> CB) {
  auto Action = [Pos, Resolve, Direction](decltype(CB) CB,
                                          Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::getTypeHierarchy(InpAST->AST, Pos, Resolve, Direction));
  };

  WorkScheduler.runWithAST("Type Hierarchy", File, Bind(Action, std::move(CB)));
}

void ClangdServer::onFileEvent(const DidChangeWatchedFilesParams &Params) {
  // FIXME: Do nothing for now. This will be used for indexing and potentially
  // invalidating other caches.
}

void ClangdServer::workspaceSymbols(
    llvm::StringRef Query, int Limit,
    Callback<std::vector<SymbolInformation>> CB) {
  std::string QueryCopy = Query;
  WorkScheduler.run(
      "getWorkspaceSymbols",
      Bind(
          [QueryCopy, Limit, this](decltype(CB) CB) {
            CB(clangd::getWorkspaceSymbols(QueryCopy, Limit, Index,
                                           WorkspaceRoot.getValueOr("")));
          },
          std::move(CB)));
}

void ClangdServer::documentSymbols(llvm::StringRef File,
                                   Callback<std::vector<DocumentSymbol>> CB) {
  auto Action = [](Callback<std::vector<DocumentSymbol>> CB,
                   llvm::Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::getDocumentSymbols(InpAST->AST));
  };
  WorkScheduler.runWithAST("documentSymbols", File,
                           Bind(Action, std::move(CB)));
}

void ClangdServer::findReferences(PathRef File, Position Pos, uint32_t Limit,
                                  Callback<std::vector<Location>> CB) {
  auto Action = [Pos, Limit, this](Callback<std::vector<Location>> CB,
                                   llvm::Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::findReferences(InpAST->AST, Pos, Limit, Index));
  };

  WorkScheduler.runWithAST("References", File, Bind(Action, std::move(CB)));
}

void ClangdServer::symbolInfo(PathRef File, Position Pos,
                              Callback<std::vector<SymbolDetails>> CB) {
  auto Action = [Pos](Callback<std::vector<SymbolDetails>> CB,
                      llvm::Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return CB(InpAST.takeError());
    CB(clangd::getSymbolInfo(InpAST->AST, Pos));
  };

  WorkScheduler.runWithAST("SymbolInfo", File, Bind(Action, std::move(CB)));
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
