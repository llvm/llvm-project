//===--- ClangdServer.cpp - Main clangd server code --------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-------------------------------------------------------------------===//

#include "ClangdServer.h"
#include "CodeComplete.h"
#include "SourceCode.h"
#include "XRefs.h"
#include "index/Merge.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Refactoring/RefactoringResultConsumer.h"
#include "clang/Tooling/Refactoring/Rename/RenamingAction.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <future>

using namespace clang;
using namespace clang::clangd;

namespace {

// Issues an async read of AST and waits for results.
template <class Ret, class Func>
Ret blockingRunWithAST(TUScheduler &S, PathRef File, Func &&F) {
  // Using shared_ptr to workaround MSVC bug. It requires future<> arguments to
  // have default and copy ctor.
  auto SharedPtrFunc = [&](llvm::Expected<InputsAndAST> Arg) {
    return std::make_shared<Ret>(F(std::move(Arg)));
  };
  std::packaged_task<std::shared_ptr<Ret>(llvm::Expected<InputsAndAST>)> Task(
      SharedPtrFunc);
  auto Future = Task.get_future();
  S.runWithAST(File, std::move(Task));
  return std::move(*Future.get());
}

// Issues an async read of preamble and waits for results.
template <class Ret, class Func>
Ret blockingRunWithPreamble(TUScheduler &S, PathRef File, Func &&F) {
  // Using shared_ptr to workaround MSVC bug. It requires future<> arguments to
  // have default and copy ctor.
  auto SharedPtrFunc = [&](llvm::Expected<InputsAndPreamble> Arg) {
    return std::make_shared<Ret>(F(std::move(Arg)));
  };
  std::packaged_task<std::shared_ptr<Ret>(llvm::Expected<InputsAndPreamble>)>
      Task(SharedPtrFunc);
  auto Future = Task.get_future();
  S.runWithPreamble(File, std::move(Task));
  return std::move(*Future.get());
}

void ignoreError(llvm::Error Err) {
  handleAllErrors(std::move(Err), [](const llvm::ErrorInfoBase &) {});
}

std::string getStandardResourceDir() {
  static int Dummy; // Just an address in this process.
  return CompilerInvocation::GetResourcesPath("clangd", (void *)&Dummy);
}

class RefactoringResultCollector final
    : public tooling::RefactoringResultConsumer {
public:
  void handleError(llvm::Error Err) override {
    assert(!Result.hasValue());
    // FIXME: figure out a way to return better message for DiagnosticError.
    // clangd uses llvm::toString to convert the Err to string, however, for
    // DiagnosticError, only "clang diagnostic" will be generated.
    Result = std::move(Err);
  }

  // Using the handle(SymbolOccurrences) from parent class.
  using tooling::RefactoringResultConsumer::handle;

  void handle(tooling::AtomicChanges SourceReplacements) override {
    assert(!Result.hasValue());
    Result = std::move(SourceReplacements);
  }

  Optional<Expected<tooling::AtomicChanges>> Result;
};

} // namespace

Tagged<IntrusiveRefCntPtr<vfs::FileSystem>>
RealFileSystemProvider::getTaggedFileSystem(PathRef File) {
  return make_tagged(vfs::getRealFileSystem(), VFSTag());
}

ClangdServer::ClangdServer(GlobalCompilationDatabase &CDB,
                           DiagnosticsConsumer &DiagConsumer,
                           FileSystemProvider &FSProvider,
                           unsigned AsyncThreadsCount,
                           bool StorePreamblesInMemory,
                           bool BuildDynamicSymbolIndex, SymbolIndex *StaticIdx,
                           llvm::Optional<StringRef> ResourceDir)
    : CompileArgs(CDB,
                  ResourceDir ? ResourceDir->str() : getStandardResourceDir()),
      DiagConsumer(DiagConsumer), FSProvider(FSProvider),
      FileIdx(BuildDynamicSymbolIndex ? new FileIndex() : nullptr),
      PCHs(std::make_shared<PCHContainerOperations>()),
      // Pass a callback into `WorkScheduler` to extract symbols from a newly
      // parsed file and rebuild the file index synchronously each time an AST
      // is parsed.
      // FIXME(ioeric): this can be slow and we may be able to index on less
      // critical paths.
      WorkScheduler(AsyncThreadsCount, StorePreamblesInMemory,
                    FileIdx
                        ? [this](PathRef Path,
                                 ParsedAST *AST) { FileIdx->update(Path, AST); }
                        : ASTParsedCallback()) {
  if (FileIdx && StaticIdx) {
    MergedIndex = mergeIndex(FileIdx.get(), StaticIdx);
    Index = MergedIndex.get();
  } else if (FileIdx)
    Index = FileIdx.get();
  else if (StaticIdx)
    Index = StaticIdx;
  else
    Index = nullptr;
}

void ClangdServer::setRootPath(PathRef RootPath) {
  std::string NewRootPath = llvm::sys::path::convert_to_slash(
      RootPath, llvm::sys::path::Style::posix);
  if (llvm::sys::fs::is_directory(NewRootPath))
    this->RootPath = NewRootPath;
}

std::future<void> ClangdServer::addDocument(PathRef File, StringRef Contents) {
  DocVersion Version = DraftMgr.updateDraft(File, Contents);
  auto TaggedFS = FSProvider.getTaggedFileSystem(File);
  return scheduleReparseAndDiags(File, VersionedDraft{Version, Contents.str()},
                                 std::move(TaggedFS));
}

std::future<void> ClangdServer::removeDocument(PathRef File) {
  DraftMgr.removeDraft(File);
  CompileArgs.invalidate(File);

  std::promise<void> DonePromise;
  std::future<void> DoneFuture = DonePromise.get_future();

  auto Callback = BindWithForward(
      [](std::promise<void> DonePromise, llvm::Error Err) {
        if (Err)
          ignoreError(std::move(Err));
        DonePromise.set_value();
      },
      std::move(DonePromise));

  WorkScheduler.remove(File, std::move(Callback));
  return DoneFuture;
}

std::future<void> ClangdServer::forceReparse(PathRef File) {
  auto FileContents = DraftMgr.getDraft(File);
  assert(FileContents.Draft &&
         "forceReparse() was called for non-added document");

  // forceReparse promises to request new compilation flags from CDB, so we
  // remove any cahced flags.
  CompileArgs.invalidate(File);

  auto TaggedFS = FSProvider.getTaggedFileSystem(File);
  return scheduleReparseAndDiags(File, std::move(FileContents),
                                 std::move(TaggedFS));
}

std::future<Tagged<CompletionList>>
ClangdServer::codeComplete(PathRef File, Position Pos,
                           const clangd::CodeCompleteOptions &Opts,
                           llvm::Optional<StringRef> OverridenContents,
                           IntrusiveRefCntPtr<vfs::FileSystem> *UsedFS) {
  std::promise<Tagged<CompletionList>> ResultPromise;
  auto Callback = [](std::promise<Tagged<CompletionList>> ResultPromise,
                     Tagged<CompletionList> Result) -> void {
    ResultPromise.set_value(std::move(Result));
  };

  auto ResultFuture = ResultPromise.get_future();
  codeComplete(File, Pos, Opts,
               BindWithForward(Callback, std::move(ResultPromise)),
               OverridenContents, UsedFS);
  return ResultFuture;
}

void ClangdServer::codeComplete(
    PathRef File, Position Pos, const clangd::CodeCompleteOptions &Opts,
    UniqueFunction<void(Tagged<CompletionList>)> Callback,
    llvm::Optional<StringRef> OverridenContents,
    IntrusiveRefCntPtr<vfs::FileSystem> *UsedFS) {
  using CallbackType = UniqueFunction<void(Tagged<CompletionList>)>;

  auto TaggedFS = FSProvider.getTaggedFileSystem(File);
  if (UsedFS)
    *UsedFS = TaggedFS.Value;

  // Copy completion options for passing them to async task handler.
  auto CodeCompleteOpts = Opts;
  if (!CodeCompleteOpts.Index) // Respect overridden index.
    CodeCompleteOpts.Index = Index;

  std::string Contents;
  if (OverridenContents) {
    Contents = OverridenContents->str();
  } else {
    VersionedDraft Latest = DraftMgr.getDraft(File);
    assert(Latest.Draft && "codeComplete called for non-added document");
    Contents = *Latest.Draft;
  }

  // Copy PCHs to avoid accessing this->PCHs concurrently
  std::shared_ptr<PCHContainerOperations> PCHs = this->PCHs;
  auto Task = [PCHs, Pos, TaggedFS, CodeCompleteOpts](
                  std::string Contents, Path File, CallbackType Callback,
                  llvm::Expected<InputsAndPreamble> IP) {
    assert(IP && "error when trying to read preamble for codeComplete");
    auto PreambleData = IP->Preamble;
    auto &Command = IP->Inputs.CompileCommand;

    // FIXME(ibiryukov): even if Preamble is non-null, we may want to check
    // both the old and the new version in case only one of them matches.
    CompletionList Result = clangd::codeComplete(
        File, Command, PreambleData ? &PreambleData->Preamble : nullptr,
        Contents, Pos, TaggedFS.Value, PCHs, CodeCompleteOpts);

    Callback(make_tagged(std::move(Result), std::move(TaggedFS.Tag)));
  };

  WorkScheduler.runWithPreamble(File, BindWithForward(Task, std::move(Contents),
                                                      File.str(),
                                                      std::move(Callback)));
}

llvm::Expected<Tagged<SignatureHelp>>
ClangdServer::signatureHelp(PathRef File, Position Pos,
                            llvm::Optional<StringRef> OverridenContents,
                            IntrusiveRefCntPtr<vfs::FileSystem> *UsedFS) {
  auto TaggedFS = FSProvider.getTaggedFileSystem(File);
  if (UsedFS)
    *UsedFS = TaggedFS.Value;

  std::string Contents;
  if (OverridenContents) {
    Contents = OverridenContents->str();
  } else {
    VersionedDraft Latest = DraftMgr.getDraft(File);
    if (!Latest.Draft)
      return llvm::make_error<llvm::StringError>(
          "signatureHelp is called for non-added document",
          llvm::errc::invalid_argument);
    Contents = std::move(*Latest.Draft);
  }

  auto Action = [=](llvm::Expected<InputsAndPreamble> IP)
      -> Expected<Tagged<SignatureHelp>> {
    if (!IP)
      return IP.takeError();
    auto PreambleData = IP->Preamble;
    auto &Command = IP->Inputs.CompileCommand;

    return make_tagged(
        clangd::signatureHelp(File, Command,
                              PreambleData ? &PreambleData->Preamble : nullptr,
                              Contents, Pos, TaggedFS.Value, PCHs),
        TaggedFS.Tag);
  };
  return blockingRunWithPreamble<Expected<Tagged<SignatureHelp>>>(WorkScheduler,
                                                                  File, Action);
}

llvm::Expected<tooling::Replacements>
ClangdServer::formatRange(StringRef Code, PathRef File, Range Rng) {
  size_t Begin = positionToOffset(Code, Rng.start);
  size_t Len = positionToOffset(Code, Rng.end) - Begin;
  return formatCode(Code, File, {tooling::Range(Begin, Len)});
}

llvm::Expected<tooling::Replacements> ClangdServer::formatFile(StringRef Code,
                                                               PathRef File) {
  // Format everything.
  return formatCode(Code, File, {tooling::Range(0, Code.size())});
}

llvm::Expected<tooling::Replacements>
ClangdServer::formatOnType(StringRef Code, PathRef File, Position Pos) {
  // Look for the previous opening brace from the character position and
  // format starting from there.
  size_t CursorPos = positionToOffset(Code, Pos);
  size_t PreviousLBracePos = StringRef(Code).find_last_of('{', CursorPos);
  if (PreviousLBracePos == StringRef::npos)
    PreviousLBracePos = CursorPos;
  size_t Len = CursorPos - PreviousLBracePos;

  return formatCode(Code, File, {tooling::Range(PreviousLBracePos, Len)});
}

Expected<std::vector<tooling::Replacement>>
ClangdServer::rename(PathRef File, Position Pos, llvm::StringRef NewName) {
  using RetType = Expected<std::vector<tooling::Replacement>>;
  auto Action = [=](Expected<InputsAndAST> InpAST) -> RetType {
    if (!InpAST)
      return InpAST.takeError();
    auto &AST = InpAST->AST;

    RefactoringResultCollector ResultCollector;
    const SourceManager &SourceMgr = AST.getASTContext().getSourceManager();
    const FileEntry *FE =
        SourceMgr.getFileEntryForID(SourceMgr.getMainFileID());
    if (!FE)
      return llvm::make_error<llvm::StringError>(
          "rename called for non-added document", llvm::errc::invalid_argument);
    SourceLocation SourceLocationBeg =
        clangd::getBeginningOfIdentifier(AST, Pos, FE);
    tooling::RefactoringRuleContext Context(
        AST.getASTContext().getSourceManager());
    Context.setASTContext(AST.getASTContext());
    auto Rename = clang::tooling::RenameOccurrences::initiate(
        Context, SourceRange(SourceLocationBeg), NewName.str());
    if (!Rename)
      return Rename.takeError();

    Rename->invoke(ResultCollector, Context);

    assert(ResultCollector.Result.hasValue());
    if (!ResultCollector.Result.getValue())
      return ResultCollector.Result->takeError();

    std::vector<tooling::Replacement> Replacements;
    for (const tooling::AtomicChange &Change : ResultCollector.Result->get()) {
      tooling::Replacements ChangeReps = Change.getReplacements();
      for (const auto &Rep : ChangeReps) {
        // FIXME: Right now we only support renaming the main file, so we
        // drop replacements not for the main file. In the future, we might
        // consider to support:
        //   * rename in any included header
        //   * rename only in the "main" header
        //   * provide an error if there are symbols we won't rename (e.g.
        //     std::vector)
        //   * rename globally in project
        //   * rename in open files
        if (Rep.getFilePath() == File)
          Replacements.push_back(Rep);
      }
    }
    return Replacements;
  };
  return blockingRunWithAST<RetType>(WorkScheduler, File, std::move(Action));
}

llvm::Optional<std::string> ClangdServer::getDocument(PathRef File) {
  auto Latest = DraftMgr.getDraft(File);
  if (!Latest.Draft)
    return llvm::None;
  return std::move(*Latest.Draft);
}

std::string ClangdServer::dumpAST(PathRef File) {
  auto Action = [](llvm::Expected<InputsAndAST> InpAST) -> std::string {
    if (!InpAST) {
      ignoreError(InpAST.takeError());
      return "<no-ast>";
    }

    std::string Result;

    llvm::raw_string_ostream ResultOS(Result);
    clangd::dumpAST(InpAST->AST, ResultOS);
    ResultOS.flush();

    return Result;
  };
  return blockingRunWithAST<std::string>(WorkScheduler, File,
                                         std::move(Action));
}

llvm::Expected<Tagged<std::vector<Location>>>
ClangdServer::findDefinitions(PathRef File, Position Pos) {
  auto TaggedFS = FSProvider.getTaggedFileSystem(File);

  using RetType = llvm::Expected<Tagged<std::vector<Location>>>;
  auto Action = [=](llvm::Expected<InputsAndAST> InpAST) -> RetType {
    if (!InpAST)
      return InpAST.takeError();
    auto Result = clangd::findDefinitions(InpAST->AST, Pos);
    return make_tagged(std::move(Result), TaggedFS.Tag);
  };
  return blockingRunWithAST<RetType>(WorkScheduler, File, Action);
}

llvm::Optional<Path> ClangdServer::switchSourceHeader(PathRef Path) {

  StringRef SourceExtensions[] = {".cpp", ".c", ".cc", ".cxx",
                                  ".c++", ".m", ".mm"};
  StringRef HeaderExtensions[] = {".h", ".hh", ".hpp", ".hxx", ".inc"};

  StringRef PathExt = llvm::sys::path::extension(Path);

  // Lookup in a list of known extensions.
  auto SourceIter =
      std::find_if(std::begin(SourceExtensions), std::end(SourceExtensions),
                   [&PathExt](PathRef SourceExt) {
                     return SourceExt.equals_lower(PathExt);
                   });
  bool IsSource = SourceIter != std::end(SourceExtensions);

  auto HeaderIter =
      std::find_if(std::begin(HeaderExtensions), std::end(HeaderExtensions),
                   [&PathExt](PathRef HeaderExt) {
                     return HeaderExt.equals_lower(PathExt);
                   });

  bool IsHeader = HeaderIter != std::end(HeaderExtensions);

  // We can only switch between extensions known extensions.
  if (!IsSource && !IsHeader)
    return llvm::None;

  // Array to lookup extensions for the switch. An opposite of where original
  // extension was found.
  ArrayRef<StringRef> NewExts;
  if (IsSource)
    NewExts = HeaderExtensions;
  else
    NewExts = SourceExtensions;

  // Storage for the new path.
  SmallString<128> NewPath = StringRef(Path);

  // Instance of vfs::FileSystem, used for file existence checks.
  auto FS = FSProvider.getTaggedFileSystem(Path).Value;

  // Loop through switched extension candidates.
  for (StringRef NewExt : NewExts) {
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

  return llvm::None;
}

llvm::Expected<tooling::Replacements>
ClangdServer::formatCode(llvm::StringRef Code, PathRef File,
                         ArrayRef<tooling::Range> Ranges) {
  // Call clang-format.
  auto TaggedFS = FSProvider.getTaggedFileSystem(File);
  auto StyleOrError =
      format::getStyle("file", File, "LLVM", Code, TaggedFS.Value.get());
  if (!StyleOrError) {
    return StyleOrError.takeError();
  } else {
    return format::reformat(StyleOrError.get(), Code, Ranges, File);
  }
}

llvm::Expected<Tagged<std::vector<DocumentHighlight>>>
ClangdServer::findDocumentHighlights(PathRef File, Position Pos) {
  auto FileContents = DraftMgr.getDraft(File);
  if (!FileContents.Draft)
    return llvm::make_error<llvm::StringError>(
        "findDocumentHighlights called on non-added file",
        llvm::errc::invalid_argument);

  auto TaggedFS = FSProvider.getTaggedFileSystem(File);

  using RetType = llvm::Expected<Tagged<std::vector<DocumentHighlight>>>;
  auto Action = [=](llvm::Expected<InputsAndAST> InpAST) -> RetType {
    if (!InpAST)
      return InpAST.takeError();
    auto Result = clangd::findDocumentHighlights(InpAST->AST, Pos);
    return make_tagged(std::move(Result), TaggedFS.Tag);
  };
  return blockingRunWithAST<RetType>(WorkScheduler, File, Action);
}

std::future<void> ClangdServer::scheduleReparseAndDiags(
    PathRef File, VersionedDraft Contents,
    Tagged<IntrusiveRefCntPtr<vfs::FileSystem>> TaggedFS) {
  tooling::CompileCommand Command = CompileArgs.getCompileCommand(File);

  using OptDiags = llvm::Optional<std::vector<DiagWithFixIts>>;

  DocVersion Version = Contents.Version;
  Path FileStr = File.str();
  VFSTag Tag = std::move(TaggedFS.Tag);

  std::promise<void> DonePromise;
  std::future<void> DoneFuture = DonePromise.get_future();

  auto Callback = [this, Version, FileStr, Tag](std::promise<void> DonePromise,
                                                OptDiags Diags) {
    auto Guard = llvm::make_scope_exit([&]() { DonePromise.set_value(); });
    if (!Diags)
      return; // A new reparse was requested before this one completed.

    // We need to serialize access to resulting diagnostics to avoid calling
    // `onDiagnosticsReady` in the wrong order.
    std::lock_guard<std::mutex> DiagsLock(DiagnosticsMutex);
    DocVersion &LastReportedDiagsVersion = ReportedDiagnosticVersions[FileStr];
    // FIXME(ibiryukov): get rid of '<' comparison here. In the current
    // implementation diagnostics will not be reported after version counters'
    // overflow. This should not happen in practice, since DocVersion is a
    // 64-bit unsigned integer.
    if (Version < LastReportedDiagsVersion)
      return;
    LastReportedDiagsVersion = Version;

    DiagConsumer.onDiagnosticsReady(
        FileStr, make_tagged(std::move(*Diags), std::move(Tag)));
  };

  WorkScheduler.update(File,
                       ParseInputs{std::move(Command),
                                   std::move(TaggedFS.Value),
                                   std::move(*Contents.Draft)},
                       BindWithForward(Callback, std::move(DonePromise)));
  return DoneFuture;
}

void ClangdServer::onFileEvent(const DidChangeWatchedFilesParams &Params) {
  // FIXME: Do nothing for now. This will be used for indexing and potentially
  // invalidating other caches.
}

std::vector<std::pair<Path, std::size_t>>
ClangdServer::getUsedBytesPerFile() const {
  return WorkScheduler.getUsedBytesPerFile();
}
