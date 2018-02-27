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
#include "Headers.h"
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

void ClangdServer::addDocument(PathRef File, StringRef Contents,
                               WantDiagnostics WantDiags) {
  DocVersion Version = DraftMgr.updateDraft(File, Contents);
  auto TaggedFS = FSProvider.getTaggedFileSystem(File);
  scheduleReparseAndDiags(File, VersionedDraft{Version, Contents.str()},
                          WantDiags, std::move(TaggedFS));
}

void ClangdServer::removeDocument(PathRef File) {
  DraftMgr.removeDraft(File);
  CompileArgs.invalidate(File);
  WorkScheduler.remove(File);
}

void ClangdServer::forceReparse(PathRef File) {
  auto FileContents = DraftMgr.getDraft(File);
  assert(FileContents.Draft &&
         "forceReparse() was called for non-added document");

  // forceReparse promises to request new compilation flags from CDB, so we
  // remove any cahced flags.
  CompileArgs.invalidate(File);

  auto TaggedFS = FSProvider.getTaggedFileSystem(File);
  scheduleReparseAndDiags(File, std::move(FileContents), WantDiagnostics::Yes,
                          std::move(TaggedFS));
}

void ClangdServer::codeComplete(
    PathRef File, Position Pos, const clangd::CodeCompleteOptions &Opts,
    UniqueFunction<void(Tagged<CompletionList>)> Callback,
    IntrusiveRefCntPtr<vfs::FileSystem> *UsedFS) {
  using CallbackType = UniqueFunction<void(Tagged<CompletionList>)>;

  auto TaggedFS = FSProvider.getTaggedFileSystem(File);
  if (UsedFS)
    *UsedFS = TaggedFS.Value;

  // Copy completion options for passing them to async task handler.
  auto CodeCompleteOpts = Opts;
  if (!CodeCompleteOpts.Index) // Respect overridden index.
    CodeCompleteOpts.Index = Index;

  VersionedDraft Latest = DraftMgr.getDraft(File);
  // FIXME(sammccall): return error for consistency?
  assert(Latest.Draft && "codeComplete called for non-added document");

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

  WorkScheduler.runWithPreamble(
      "CodeComplete", File,
      Bind(Task, std::move(*Latest.Draft), File.str(), std::move(Callback)));
}

void ClangdServer::signatureHelp(
    PathRef File, Position Pos,
    UniqueFunction<void(llvm::Expected<Tagged<SignatureHelp>>)> Callback,
    IntrusiveRefCntPtr<vfs::FileSystem> *UsedFS) {
  auto TaggedFS = FSProvider.getTaggedFileSystem(File);
  if (UsedFS)
    *UsedFS = TaggedFS.Value;

  VersionedDraft Latest = DraftMgr.getDraft(File);
  if (!Latest.Draft)
    return Callback(llvm::make_error<llvm::StringError>(
        "signatureHelp is called for non-added document",
        llvm::errc::invalid_argument));

  auto PCHs = this->PCHs;
  auto Action = [Pos, TaggedFS, PCHs](std::string Contents, Path File,
                                      decltype(Callback) Callback,
                                      llvm::Expected<InputsAndPreamble> IP) {
    if (!IP)
      return Callback(IP.takeError());

    auto PreambleData = IP->Preamble;
    auto &Command = IP->Inputs.CompileCommand;
    Callback(make_tagged(
        clangd::signatureHelp(File, Command,
                              PreambleData ? &PreambleData->Preamble : nullptr,
                              Contents, Pos, TaggedFS.Value, PCHs),
        TaggedFS.Tag));
  };

  WorkScheduler.runWithPreamble(
      "SignatureHelp", File,
      Bind(Action, std::move(*Latest.Draft), File.str(), std::move(Callback)));
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

void ClangdServer::rename(
    PathRef File, Position Pos, llvm::StringRef NewName,
    UniqueFunction<void(Expected<std::vector<tooling::Replacement>>)>
        Callback) {
  auto Action = [Pos](Path File, std::string NewName,
                      decltype(Callback) Callback,
                      Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return Callback(InpAST.takeError());
    auto &AST = InpAST->AST;

    RefactoringResultCollector ResultCollector;
    const SourceManager &SourceMgr = AST.getASTContext().getSourceManager();
    const FileEntry *FE =
        SourceMgr.getFileEntryForID(SourceMgr.getMainFileID());
    if (!FE)
      return Callback(llvm::make_error<llvm::StringError>(
          "rename called for non-added document",
          llvm::errc::invalid_argument));
    SourceLocation SourceLocationBeg =
        clangd::getBeginningOfIdentifier(AST, Pos, FE);
    tooling::RefactoringRuleContext Context(
        AST.getASTContext().getSourceManager());
    Context.setASTContext(AST.getASTContext());
    auto Rename = clang::tooling::RenameOccurrences::initiate(
        Context, SourceRange(SourceLocationBeg), NewName);
    if (!Rename)
      return Callback(Rename.takeError());

    Rename->invoke(ResultCollector, Context);

    assert(ResultCollector.Result.hasValue());
    if (!ResultCollector.Result.getValue())
      return Callback(ResultCollector.Result->takeError());

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
    return Callback(Replacements);
  };

  WorkScheduler.runWithAST(
      "Rename", File,
      Bind(Action, File.str(), NewName.str(), std::move(Callback)));
}

/// Creates a `HeaderFile` from \p Header which can be either a URI or a literal
/// include.
static llvm::Expected<HeaderFile> toHeaderFile(StringRef Header,
                                               llvm::StringRef HintPath) {
  if (isLiteralInclude(Header))
    return HeaderFile{Header.str(), /*Verbatim=*/true};
  auto U = URI::parse(Header);
  if (!U)
    return U.takeError();
  auto Resolved = URI::resolve(*U, HintPath);
  if (!Resolved)
    return Resolved.takeError();
  return HeaderFile{std::move(*Resolved), /*Verbatim=*/false};
};

Expected<tooling::Replacements>
ClangdServer::insertInclude(PathRef File, StringRef Code,
                            StringRef DeclaringHeader,
                            StringRef InsertedHeader) {
  assert(!DeclaringHeader.empty() && !InsertedHeader.empty());
  std::string ToInclude;
  auto ResolvedOrginal = toHeaderFile(DeclaringHeader, File);
  if (!ResolvedOrginal)
    return ResolvedOrginal.takeError();
  auto ResolvedPreferred = toHeaderFile(InsertedHeader, File);
  if (!ResolvedPreferred)
    return ResolvedPreferred.takeError();
  tooling::CompileCommand CompileCommand = CompileArgs.getCompileCommand(File);
  auto Include = calculateIncludePath(
      File, Code, *ResolvedOrginal, *ResolvedPreferred, CompileCommand,
      FSProvider.getTaggedFileSystem(File).Value);
  if (!Include)
    return Include.takeError();
  if (Include->empty())
    return tooling::Replacements();
  ToInclude = std::move(*Include);

  auto Style = format::getStyle("file", File, "llvm");
  if (!Style) {
    llvm::consumeError(Style.takeError());
    // FIXME(ioeric): needs more consistent style support in clangd server.
    Style = format::getLLVMStyle();
  }
  // Replacement with offset UINT_MAX and length 0 will be treated as include
  // insertion.
  tooling::Replacement R(File, /*Offset=*/UINT_MAX, 0, "#include " + ToInclude);
  return format::cleanupAroundReplacements(Code, tooling::Replacements(R),
                                           *Style);
}

llvm::Optional<std::string> ClangdServer::getDocument(PathRef File) {
  auto Latest = DraftMgr.getDraft(File);
  if (!Latest.Draft)
    return llvm::None;
  return std::move(*Latest.Draft);
}

void ClangdServer::dumpAST(PathRef File,
                           UniqueFunction<void(std::string)> Callback) {
  auto Action = [](decltype(Callback) Callback,
                   llvm::Expected<InputsAndAST> InpAST) {
    if (!InpAST) {
      ignoreError(InpAST.takeError());
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

void ClangdServer::findDefinitions(
    PathRef File, Position Pos,
    UniqueFunction<void(llvm::Expected<Tagged<std::vector<Location>>>)>
        Callback) {
  auto TaggedFS = FSProvider.getTaggedFileSystem(File);
  auto Action = [Pos, TaggedFS](decltype(Callback) Callback,
                                llvm::Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return Callback(InpAST.takeError());
    auto Result = clangd::findDefinitions(InpAST->AST, Pos);
    Callback(make_tagged(std::move(Result), TaggedFS.Tag));
  };

  WorkScheduler.runWithAST("Definitions", File,
                           Bind(Action, std::move(Callback)));
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

void ClangdServer::findDocumentHighlights(
    PathRef File, Position Pos,
    UniqueFunction<void(llvm::Expected<Tagged<std::vector<DocumentHighlight>>>)>
        Callback) {
  auto FileContents = DraftMgr.getDraft(File);
  if (!FileContents.Draft)
    return Callback(llvm::make_error<llvm::StringError>(
        "findDocumentHighlights called on non-added file",
        llvm::errc::invalid_argument));

  auto TaggedFS = FSProvider.getTaggedFileSystem(File);

  auto Action = [TaggedFS, Pos](decltype(Callback) Callback,
                                llvm::Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return Callback(InpAST.takeError());
    auto Result = clangd::findDocumentHighlights(InpAST->AST, Pos);
    Callback(make_tagged(std::move(Result), TaggedFS.Tag));
  };

  WorkScheduler.runWithAST("Highlights", File,
                           Bind(Action, std::move(Callback)));
}

void ClangdServer::findHover(
    PathRef File, Position Pos,
    UniqueFunction<void(llvm::Expected<Tagged<Hover>>)> Callback) {
  Hover FinalHover;
  auto FileContents = DraftMgr.getDraft(File);
  if (!FileContents.Draft)
    return Callback(llvm::make_error<llvm::StringError>(
        "findHover called on non-added file", llvm::errc::invalid_argument));

  auto TaggedFS = FSProvider.getTaggedFileSystem(File);

  auto Action = [Pos, TaggedFS](decltype(Callback) Callback,
                                llvm::Expected<InputsAndAST> InpAST) {
    if (!InpAST)
      return Callback(InpAST.takeError());

    Hover Result = clangd::getHover(InpAST->AST, Pos);
    Callback(make_tagged(std::move(Result), TaggedFS.Tag));
  };

  WorkScheduler.runWithAST("Hover", File, Bind(Action, std::move(Callback)));
}

void ClangdServer::scheduleReparseAndDiags(
    PathRef File, VersionedDraft Contents, WantDiagnostics WantDiags,
    Tagged<IntrusiveRefCntPtr<vfs::FileSystem>> TaggedFS) {
  tooling::CompileCommand Command = CompileArgs.getCompileCommand(File);

  DocVersion Version = Contents.Version;
  Path FileStr = File.str();
  VFSTag Tag = std::move(TaggedFS.Tag);

  auto Callback = [this, Version, FileStr,
                   Tag](std::vector<DiagWithFixIts> Diags) {
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
        FileStr, make_tagged(std::move(Diags), std::move(Tag)));
  };

  WorkScheduler.update(File,
                       ParseInputs{std::move(Command),
                                   std::move(TaggedFS.Value),
                                   std::move(*Contents.Draft)},
                       WantDiags, std::move(Callback));
}

void ClangdServer::reparseOpenedFiles() {
  for (const Path &FilePath : DraftMgr.getActiveFiles())
    forceReparse(FilePath);
}

void ClangdServer::onFileEvent(const DidChangeWatchedFilesParams &Params) {
  // FIXME: Do nothing for now. This will be used for indexing and potentially
  // invalidating other caches.
}

std::vector<std::pair<Path, std::size_t>>
ClangdServer::getUsedBytesPerFile() const {
  return WorkScheduler.getUsedBytesPerFile();
}

LLVM_NODISCARD bool
ClangdServer::blockUntilIdleForTest(llvm::Optional<double> TimeoutSeconds) {
  return WorkScheduler.blockUntilIdle(timeoutSeconds(TimeoutSeconds));
}
