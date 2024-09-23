//===--- Preamble.cpp - Reusing expensive parts of the AST ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Preamble.h"
#include "CollectMacros.h"
#include "Compiler.h"
#include "Config.h"
#include "Diagnostics.h"
#include "FS.h"
#include "FeatureModule.h"
#include "Headers.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "clang-include-cleaner/Record.h"
#include "support/Logger.h"
#include "support/Path.h"
#include "support/ThreadsafeFS.h"
#include "support/Trace.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticLex.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/PrecompiledPreamble.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

namespace clang {
namespace clangd {
namespace {

bool compileCommandsAreEqual(const tooling::CompileCommand &LHS,
                             const tooling::CompileCommand &RHS) {
  // We don't check for Output, it should not matter to clangd.
  return LHS.Directory == RHS.Directory && LHS.Filename == RHS.Filename &&
         llvm::ArrayRef(LHS.CommandLine).equals(RHS.CommandLine);
}

class CppFilePreambleCallbacks : public PreambleCallbacks {
public:
  CppFilePreambleCallbacks(
      PathRef File, PreambleBuildStats *Stats, bool ParseForwardingFunctions,
      std::function<void(CompilerInstance &)> BeforeExecuteCallback)
      : File(File), Stats(Stats),
        ParseForwardingFunctions(ParseForwardingFunctions),
        BeforeExecuteCallback(std::move(BeforeExecuteCallback)) {}

  IncludeStructure takeIncludes() { return std::move(Includes); }

  MainFileMacros takeMacros() { return std::move(Macros); }

  std::vector<PragmaMark> takeMarks() { return std::move(Marks); }

  include_cleaner::PragmaIncludes takePragmaIncludes() {
    return std::move(Pragmas);
  }

  std::optional<CapturedASTCtx> takeLife() { return std::move(CapturedCtx); }

  bool isMainFileIncludeGuarded() const { return IsMainFileIncludeGuarded; }

  void AfterExecute(CompilerInstance &CI) override {
    // As part of the Preamble compilation, ASTConsumer
    // PrecompilePreambleConsumer/PCHGenerator is setup. This would be called
    // when Preamble consists of modules. Therefore while capturing AST context,
    // we have to reset ast consumer and ASTMutationListener.
    if (CI.getASTReader()) {
      CI.getASTReader()->setDeserializationListener(nullptr);
      // This just sets consumer to null when DeserializationListener is null.
      CI.getASTReader()->StartTranslationUnit(nullptr);
    }
    CI.getASTContext().setASTMutationListener(nullptr);
    CapturedCtx.emplace(CI);

    const SourceManager &SM = CI.getSourceManager();
    OptionalFileEntryRef MainFE = SM.getFileEntryRefForID(SM.getMainFileID());
    IsMainFileIncludeGuarded =
        CI.getPreprocessor().getHeaderSearchInfo().isFileMultipleIncludeGuarded(
            *MainFE);

    if (Stats) {
      const ASTContext &AST = CI.getASTContext();
      Stats->BuildSize = AST.getASTAllocatedMemory();
      Stats->BuildSize += AST.getSideTableAllocatedMemory();
      Stats->BuildSize += AST.Idents.getAllocator().getTotalMemory();
      Stats->BuildSize += AST.Selectors.getTotalMemory();

      Stats->BuildSize += AST.getSourceManager().getContentCacheSize();
      Stats->BuildSize += AST.getSourceManager().getDataStructureSizes();
      Stats->BuildSize +=
          AST.getSourceManager().getMemoryBufferSizes().malloc_bytes;

      const Preprocessor &PP = CI.getPreprocessor();
      Stats->BuildSize += PP.getTotalMemory();
      if (PreprocessingRecord *PRec = PP.getPreprocessingRecord())
        Stats->BuildSize += PRec->getTotalMemory();
      Stats->BuildSize += PP.getHeaderSearchInfo().getTotalMemory();
    }
  }

  void BeforeExecute(CompilerInstance &CI) override {
    LangOpts = &CI.getLangOpts();
    SourceMgr = &CI.getSourceManager();
    PP = &CI.getPreprocessor();
    Includes.collect(CI);
    Pragmas.record(CI);
    if (BeforeExecuteCallback)
      BeforeExecuteCallback(CI);
  }

  std::unique_ptr<PPCallbacks> createPPCallbacks() override {
    assert(SourceMgr && LangOpts && PP &&
           "SourceMgr, LangOpts and PP must be set at this point");

    return std::make_unique<PPChainedCallbacks>(
        std::make_unique<CollectMainFileMacros>(*PP, Macros),
        collectPragmaMarksCallback(*SourceMgr, Marks));
  }

  static bool isLikelyForwardingFunction(FunctionTemplateDecl *FT) {
    const auto *FD = FT->getTemplatedDecl();
    const auto NumParams = FD->getNumParams();
    // Check whether its last parameter is a parameter pack...
    if (NumParams > 0) {
      const auto *LastParam = FD->getParamDecl(NumParams - 1);
      if (const auto *PET = dyn_cast<PackExpansionType>(LastParam->getType())) {
        // ... of the type T&&... or T...
        const auto BaseType = PET->getPattern().getNonReferenceType();
        if (const auto *TTPT =
                dyn_cast<TemplateTypeParmType>(BaseType.getTypePtr())) {
          // ... whose template parameter comes from the function directly
          if (FT->getTemplateParameters()->getDepth() == TTPT->getDepth()) {
            return true;
          }
        }
      }
    }
    return false;
  }

  bool shouldSkipFunctionBody(Decl *D) override {
    // Usually we don't need to look inside the bodies of header functions
    // to understand the program. However when forwarding function like
    // emplace() forward their arguments to some other function, the
    // interesting overload resolution happens inside the forwarding
    // function's body. To provide more meaningful diagnostics,
    // code completion, and parameter hints we should parse (and later
    // instantiate) the bodies.
    if (auto *FT = llvm::dyn_cast<clang::FunctionTemplateDecl>(D)) {
      if (ParseForwardingFunctions) {
        // Don't skip parsing the body if it looks like a forwarding function
        if (isLikelyForwardingFunction(FT))
          return false;
      } else {
        // By default, only take care of make_unique
        // std::make_unique is trivial, and we diagnose bad constructor calls.
        if (const auto *II = FT->getDeclName().getAsIdentifierInfo()) {
          if (II->isStr("make_unique") && FT->isInStdNamespace())
            return false;
        }
      }
    }
    return true;
  }

private:
  PathRef File;
  IncludeStructure Includes;
  include_cleaner::PragmaIncludes Pragmas;
  MainFileMacros Macros;
  std::vector<PragmaMark> Marks;
  bool IsMainFileIncludeGuarded = false;
  const clang::LangOptions *LangOpts = nullptr;
  const SourceManager *SourceMgr = nullptr;
  const Preprocessor *PP = nullptr;
  PreambleBuildStats *Stats;
  bool ParseForwardingFunctions;
  std::function<void(CompilerInstance &)> BeforeExecuteCallback;
  std::optional<CapturedASTCtx> CapturedCtx;
};

// Represents directives other than includes, where basic textual information is
// enough.
struct TextualPPDirective {
  unsigned DirectiveLine;
  // Full text that's representing the directive, including the `#`.
  std::string Text;
  unsigned Offset;
  tok::PPKeywordKind Directive = tok::PPKeywordKind::pp_not_keyword;
  // Name of the macro being defined in the case of a #define directive.
  std::string MacroName;

  bool operator==(const TextualPPDirective &RHS) const {
    return std::tie(DirectiveLine, Offset, Text) ==
           std::tie(RHS.DirectiveLine, RHS.Offset, RHS.Text);
  }
};

// Formats a PP directive consisting of Prefix (e.g. "#define ") and Body ("X
// 10"). The formatting is copied so that the tokens in Body have PresumedLocs
// with correct columns and lines.
std::string spellDirective(llvm::StringRef Prefix,
                           CharSourceRange DirectiveRange,
                           const LangOptions &LangOpts, const SourceManager &SM,
                           unsigned &DirectiveLine, unsigned &Offset) {
  std::string SpelledDirective;
  llvm::raw_string_ostream OS(SpelledDirective);
  OS << Prefix;

  // Make sure DirectiveRange is a char range and doesn't contain macro ids.
  DirectiveRange = SM.getExpansionRange(DirectiveRange);
  if (DirectiveRange.isTokenRange()) {
    DirectiveRange.setEnd(
        Lexer::getLocForEndOfToken(DirectiveRange.getEnd(), 0, SM, LangOpts));
  }

  auto DecompLoc = SM.getDecomposedLoc(DirectiveRange.getBegin());
  DirectiveLine = SM.getLineNumber(DecompLoc.first, DecompLoc.second);
  Offset = DecompLoc.second;
  auto TargetColumn = SM.getColumnNumber(DecompLoc.first, DecompLoc.second) - 1;

  // Pad with spaces before DirectiveRange to make sure it will be on right
  // column when patched.
  if (Prefix.size() <= TargetColumn) {
    // There is enough space for Prefix and space before directive, use it.
    // We try to squeeze the Prefix into the same line whenever we can, as
    // putting onto a separate line won't work at the beginning of the file.
    OS << std::string(TargetColumn - Prefix.size(), ' ');
  } else {
    // Prefix was longer than the space we had. We produce e.g.:
    // #line N-1
    // #define \
    //    X 10
    OS << "\\\n" << std::string(TargetColumn, ' ');
    // Decrement because we put an additional line break before
    // DirectiveRange.begin().
    --DirectiveLine;
  }
  OS << toSourceCode(SM, DirectiveRange.getAsRange());
  return OS.str();
}

// Collects #define directives inside the main file.
struct DirectiveCollector : public PPCallbacks {
  DirectiveCollector(const Preprocessor &PP,
                     std::vector<TextualPPDirective> &TextualDirectives)
      : LangOpts(PP.getLangOpts()), SM(PP.getSourceManager()),
        TextualDirectives(TextualDirectives) {}

  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID) override {
    InMainFile = SM.isWrittenInMainFile(Loc);
  }

  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override {
    if (!InMainFile)
      return;
    TextualDirectives.emplace_back();
    TextualPPDirective &TD = TextualDirectives.back();
    TD.Directive = tok::pp_define;
    TD.MacroName = MacroNameTok.getIdentifierInfo()->getName().str();

    const auto *MI = MD->getMacroInfo();
    TD.Text =
        spellDirective("#define ",
                       CharSourceRange::getTokenRange(
                           MI->getDefinitionLoc(), MI->getDefinitionEndLoc()),
                       LangOpts, SM, TD.DirectiveLine, TD.Offset);
  }

private:
  bool InMainFile = true;
  const LangOptions &LangOpts;
  const SourceManager &SM;
  std::vector<TextualPPDirective> &TextualDirectives;
};

struct ScannedPreamble {
  std::vector<Inclusion> Includes;
  std::vector<TextualPPDirective> TextualDirectives;
  // Literal lines of the preamble contents.
  std::vector<llvm::StringRef> Lines;
  PreambleBounds Bounds = {0, false};
  std::vector<PragmaMark> Marks;
  MainFileMacros Macros;
};

/// Scans the preprocessor directives in the preamble section of the file by
/// running preprocessor over \p Contents. Returned includes do not contain
/// resolved paths. \p Cmd is used to build the compiler invocation, which might
/// stat/read files.
llvm::Expected<ScannedPreamble>
scanPreamble(llvm::StringRef Contents, const tooling::CompileCommand &Cmd) {
  class EmptyFS : public ThreadsafeFS {
  private:
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> viewImpl() const override {
      return new llvm::vfs::InMemoryFileSystem;
    }
  };
  EmptyFS FS;
  // Build and run Preprocessor over the preamble.
  ParseInputs PI;
  // Memory buffers below expect null-terminated && non-null strings. So make
  // sure to always use PI.Contents!
  PI.Contents = Contents.str();
  PI.TFS = &FS;
  PI.CompileCommand = Cmd;
  IgnoringDiagConsumer IgnoreDiags;
  auto CI = buildCompilerInvocation(PI, IgnoreDiags);
  if (!CI)
    return error("failed to create compiler invocation");
  CI->getDiagnosticOpts().IgnoreWarnings = true;
  auto ContentsBuffer = llvm::MemoryBuffer::getMemBuffer(PI.Contents);
  // This means we're scanning (though not preprocessing) the preamble section
  // twice. However, it's important to precisely follow the preamble bounds used
  // elsewhere.
  auto Bounds = ComputePreambleBounds(CI->getLangOpts(), *ContentsBuffer, 0);
  auto PreambleContents = llvm::MemoryBuffer::getMemBufferCopy(
      llvm::StringRef(PI.Contents).take_front(Bounds.Size));
  auto Clang = prepareCompilerInstance(
      std::move(CI), nullptr, std::move(PreambleContents),
      // Provide an empty FS to prevent preprocessor from performing IO. This
      // also implies missing resolved paths for includes.
      FS.view(std::nullopt), IgnoreDiags);
  if (Clang->getFrontendOpts().Inputs.empty())
    return error("compiler instance had no inputs");
  // We are only interested in main file includes.
  Clang->getPreprocessorOpts().SingleFileParseMode = true;
  Clang->getPreprocessorOpts().UsePredefines = false;
  PreprocessOnlyAction Action;
  if (!Action.BeginSourceFile(*Clang, Clang->getFrontendOpts().Inputs[0]))
    return error("failed BeginSourceFile");
  Preprocessor &PP = Clang->getPreprocessor();
  const auto &SM = PP.getSourceManager();
  IncludeStructure Includes;
  Includes.collect(*Clang);
  ScannedPreamble SP;
  SP.Bounds = Bounds;
  PP.addPPCallbacks(
      std::make_unique<DirectiveCollector>(PP, SP.TextualDirectives));
  PP.addPPCallbacks(collectPragmaMarksCallback(SM, SP.Marks));
  PP.addPPCallbacks(std::make_unique<CollectMainFileMacros>(PP, SP.Macros));
  if (llvm::Error Err = Action.Execute())
    return std::move(Err);
  Action.EndSourceFile();
  SP.Includes = std::move(Includes.MainFileIncludes);
  llvm::append_range(SP.Lines, llvm::split(Contents, "\n"));
  return SP;
}

const char *spellingForIncDirective(tok::PPKeywordKind IncludeDirective) {
  switch (IncludeDirective) {
  case tok::pp_include:
    return "include";
  case tok::pp_import:
    return "import";
  case tok::pp_include_next:
    return "include_next";
  default:
    break;
  }
  llvm_unreachable("not an include directive");
}

// Accumulating wall time timer. Similar to llvm::Timer, but much cheaper,
// it only tracks wall time.
// Since this is a generic timer, We may want to move this to support/ if we
// find a use case outside of FS time tracking.
class WallTimer {
public:
  WallTimer() : TotalTime(std::chrono::steady_clock::duration::zero()) {}
  // [Re-]Start the timer.
  void startTimer() { StartTime = std::chrono::steady_clock::now(); }
  // Stop the timer and update total time.
  void stopTimer() {
    TotalTime += std::chrono::steady_clock::now() - StartTime;
  }
  // Return total time, in seconds.
  double getTime() { return std::chrono::duration<double>(TotalTime).count(); }

private:
  std::chrono::steady_clock::duration TotalTime;
  std::chrono::steady_clock::time_point StartTime;
};

class WallTimerRegion {
public:
  WallTimerRegion(WallTimer &T) : T(T) { T.startTimer(); }
  ~WallTimerRegion() { T.stopTimer(); }

private:
  WallTimer &T;
};

// Used by TimerFS, tracks time spent in status() and getBuffer() calls while
// proxying to underlying File implementation.
class TimerFile : public llvm::vfs::File {
public:
  TimerFile(WallTimer &Timer, std::unique_ptr<File> InnerFile)
      : Timer(Timer), InnerFile(std::move(InnerFile)) {}

  llvm::ErrorOr<llvm::vfs::Status> status() override {
    WallTimerRegion T(Timer);
    return InnerFile->status();
  }
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  getBuffer(const Twine &Name, int64_t FileSize, bool RequiresNullTerminator,
            bool IsVolatile) override {
    WallTimerRegion T(Timer);
    return InnerFile->getBuffer(Name, FileSize, RequiresNullTerminator,
                                IsVolatile);
  }
  std::error_code close() override {
    WallTimerRegion T(Timer);
    return InnerFile->close();
  }

private:
  WallTimer &Timer;
  std::unique_ptr<llvm::vfs::File> InnerFile;
};

// A wrapper for FileSystems that tracks the amount of time spent in status()
// and openFileForRead() calls.
class TimerFS : public llvm::vfs::ProxyFileSystem {
public:
  TimerFS(llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS)
      : ProxyFileSystem(std::move(FS)) {}

  llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
  openFileForRead(const llvm::Twine &Path) override {
    WallTimerRegion T(Timer);
    auto FileOr = getUnderlyingFS().openFileForRead(Path);
    if (!FileOr)
      return FileOr;
    return std::make_unique<TimerFile>(Timer, std::move(FileOr.get()));
  }

  llvm::ErrorOr<llvm::vfs::Status> status(const llvm::Twine &Path) override {
    WallTimerRegion T(Timer);
    return getUnderlyingFS().status(Path);
  }

  double getTime() { return Timer.getTime(); }

private:
  WallTimer Timer;
};

// Helpers for patching diagnostics between two versions of file contents.
class DiagPatcher {
  llvm::ArrayRef<llvm::StringRef> OldLines;
  llvm::ArrayRef<llvm::StringRef> CurrentLines;
  llvm::StringMap<llvm::SmallVector<int>> CurrentContentsToLine;

  // Translates a range from old lines to current lines.
  // Finds the consecutive set of lines that corresponds to the same contents in
  // old and current, and applies the same translation to the range.
  // Returns true if translation succeeded.
  bool translateRange(Range &R) {
    int OldStart = R.start.line;
    int OldEnd = R.end.line;
    assert(OldStart <= OldEnd);

    size_t RangeLen = OldEnd - OldStart + 1;
    auto RangeContents = OldLines.slice(OldStart).take_front(RangeLen);
    // Make sure the whole range is covered in old contents.
    if (RangeContents.size() < RangeLen)
      return false;

    std::optional<int> Closest;
    for (int AlternateLine : CurrentContentsToLine.lookup(RangeContents[0])) {
      // Check if AlternateLine matches all lines in the range.
      if (RangeContents !=
          CurrentLines.slice(AlternateLine).take_front(RangeLen))
        continue;
      int Delta = AlternateLine - OldStart;
      if (!Closest.has_value() || abs(Delta) < abs(*Closest))
        Closest = Delta;
    }
    // Couldn't find any viable matches in the current contents.
    if (!Closest.has_value())
      return false;
    R.start.line += *Closest;
    R.end.line += *Closest;
    return true;
  }

  // Translates a Note by patching its range when inside main file. Returns true
  // on success.
  bool translateNote(Note &N) {
    if (!N.InsideMainFile)
      return true;
    if (translateRange(N.Range))
      return true;
    return false;
  }

  // Tries to translate all the edit ranges inside the fix. Returns true on
  // success. On failure fixes might be in an invalid state.
  bool translateFix(Fix &F) {
    return llvm::all_of(
        F.Edits, [this](TextEdit &E) { return translateRange(E.range); });
  }

public:
  DiagPatcher(llvm::ArrayRef<llvm::StringRef> OldLines,
              llvm::ArrayRef<llvm::StringRef> CurrentLines) {
    this->OldLines = OldLines;
    this->CurrentLines = CurrentLines;
    for (int Line = 0, E = CurrentLines.size(); Line != E; ++Line) {
      llvm::StringRef Contents = CurrentLines[Line];
      CurrentContentsToLine[Contents].push_back(Line);
    }
  }
  // Translate diagnostic by moving its main range to new location (if inside
  // the main file). Preserve all the notes and fixes that can be translated to
  // new contents.
  // Drops the whole diagnostic if main range can't be patched.
  std::optional<Diag> translateDiag(const Diag &D) {
    Range NewRange = D.Range;
    // Patch range if it's inside main file.
    if (D.InsideMainFile && !translateRange(NewRange)) {
      // Drop the diagnostic if we couldn't patch the range.
      return std::nullopt;
    }

    Diag NewD = D;
    NewD.Range = NewRange;
    // Translate ranges inside notes and fixes too, dropping the ones that are
    // no longer relevant.
    llvm::erase_if(NewD.Notes, [this](Note &N) { return !translateNote(N); });
    llvm::erase_if(NewD.Fixes, [this](Fix &F) { return !translateFix(F); });
    return NewD;
  }
};
} // namespace

std::shared_ptr<const PreambleData>
buildPreamble(PathRef FileName, CompilerInvocation CI,
              const ParseInputs &Inputs, bool StoreInMemory,
              PreambleParsedCallback PreambleCallback,
              PreambleBuildStats *Stats) {
  // Note that we don't need to copy the input contents, preamble can live
  // without those.
  auto ContentsBuffer =
      llvm::MemoryBuffer::getMemBuffer(Inputs.Contents, FileName);
  auto Bounds = ComputePreambleBounds(CI.getLangOpts(), *ContentsBuffer, 0);

  trace::Span Tracer("BuildPreamble");
  SPAN_ATTACH(Tracer, "File", FileName);
  std::vector<std::unique_ptr<FeatureModule::ASTListener>> ASTListeners;
  if (Inputs.FeatureModules) {
    for (auto &M : *Inputs.FeatureModules) {
      if (auto Listener = M.astListeners())
        ASTListeners.emplace_back(std::move(Listener));
    }
  }
  StoreDiags PreambleDiagnostics;
  PreambleDiagnostics.setDiagCallback(
      [&ASTListeners](const clang::Diagnostic &D, clangd::Diag &Diag) {
        for (const auto &L : ASTListeners)
          L->sawDiagnostic(D, Diag);
      });
  llvm::IntrusiveRefCntPtr<DiagnosticsEngine> PreambleDiagsEngine =
      CompilerInstance::createDiagnostics(&CI.getDiagnosticOpts(),
                                          &PreambleDiagnostics,
                                          /*ShouldOwnClient=*/false);
  const Config &Cfg = Config::current();
  PreambleDiagnostics.setLevelAdjuster([&](DiagnosticsEngine::Level DiagLevel,
                                           const clang::Diagnostic &Info) {
    if (Cfg.Diagnostics.SuppressAll ||
        isDiagnosticSuppressed(Info, Cfg.Diagnostics.Suppress,
                               CI.getLangOpts()))
      return DiagnosticsEngine::Ignored;
    switch (Info.getID()) {
    case diag::warn_no_newline_eof:
    case diag::warn_cxx98_compat_no_newline_eof:
    case diag::ext_no_newline_eof:
      // If the preamble doesn't span the whole file, drop the no newline at
      // eof warnings.
      return Bounds.Size != ContentsBuffer->getBufferSize()
                 ? DiagnosticsEngine::Level::Ignored
                 : DiagLevel;
    }
    return DiagLevel;
  });

  // Skip function bodies when building the preamble to speed up building
  // the preamble and make it smaller.
  assert(!CI.getFrontendOpts().SkipFunctionBodies);
  CI.getFrontendOpts().SkipFunctionBodies = true;
  // We don't want to write comment locations into PCH. They are racy and slow
  // to read back. We rely on dynamic index for the comments instead.
  CI.getPreprocessorOpts().WriteCommentListToPCH = false;

  CppFilePreambleCallbacks CapturedInfo(
      FileName, Stats, Inputs.Opts.PreambleParseForwardingFunctions,
      [&ASTListeners](CompilerInstance &CI) {
        for (const auto &L : ASTListeners)
          L->beforeExecute(CI);
      });
  auto VFS = Inputs.TFS->view(Inputs.CompileCommand.Directory);
  llvm::SmallString<32> AbsFileName(FileName);
  VFS->makeAbsolute(AbsFileName);
  auto StatCache = std::make_shared<PreambleFileStatusCache>(AbsFileName);
  auto StatCacheFS = StatCache->getProducingFS(VFS);
  llvm::IntrusiveRefCntPtr<TimerFS> TimedFS(new TimerFS(StatCacheFS));

  WallTimer PreambleTimer;
  PreambleTimer.startTimer();
  auto BuiltPreamble = PrecompiledPreamble::Build(
      CI, ContentsBuffer.get(), Bounds, *PreambleDiagsEngine,
      Stats ? TimedFS : StatCacheFS, std::make_shared<PCHContainerOperations>(),
      StoreInMemory, /*StoragePath=*/"", CapturedInfo);

  PreambleTimer.stopTimer();

  // We have to setup DiagnosticConsumer that will be alife
  // while preamble callback is executed
  PreambleDiagsEngine->setClient(new IgnoringDiagConsumer, true);
  // Reset references to ref-counted-ptrs before executing the callbacks, to
  // prevent resetting them concurrently.
  PreambleDiagsEngine.reset();
  CI.DiagnosticOpts.reset();

  // When building the AST for the main file, we do want the function
  // bodies.
  CI.getFrontendOpts().SkipFunctionBodies = false;

  if (Stats != nullptr) {
    Stats->TotalBuildTime = PreambleTimer.getTime();
    Stats->FileSystemTime = TimedFS->getTime();
    Stats->SerializedSize = BuiltPreamble ? BuiltPreamble->getSize() : 0;
  }

  if (BuiltPreamble) {
    log("Built preamble of size {0} for file {1} version {2} in {3} seconds",
        BuiltPreamble->getSize(), FileName, Inputs.Version,
        PreambleTimer.getTime());
    std::vector<Diag> Diags = PreambleDiagnostics.take();
    auto Result = std::make_shared<PreambleData>(std::move(*BuiltPreamble));
    Result->Version = Inputs.Version;
    Result->CompileCommand = Inputs.CompileCommand;
    Result->Diags = std::move(Diags);
    Result->Includes = CapturedInfo.takeIncludes();
    Result->Pragmas = std::make_shared<const include_cleaner::PragmaIncludes>(
        CapturedInfo.takePragmaIncludes());

    if (Inputs.ModulesManager) {
      WallTimer PrerequisiteModuleTimer;
      PrerequisiteModuleTimer.startTimer();
      Result->RequiredModules =
          Inputs.ModulesManager->buildPrerequisiteModulesFor(FileName,
                                                             *Inputs.TFS);
      PrerequisiteModuleTimer.stopTimer();

      log("Built prerequisite modules for file {0} in {1} seconds", FileName,
          PrerequisiteModuleTimer.getTime());
    }

    Result->Macros = CapturedInfo.takeMacros();
    Result->Marks = CapturedInfo.takeMarks();
    Result->StatCache = StatCache;
    Result->MainIsIncludeGuarded = CapturedInfo.isMainFileIncludeGuarded();
    Result->TargetOpts = CI.TargetOpts;
    if (PreambleCallback) {
      trace::Span Tracer("Running PreambleCallback");
      auto Ctx = CapturedInfo.takeLife();
      // Stat cache is thread safe only when there are no producers. Hence
      // change the VFS underneath to a consuming fs.
      Ctx->getFileManager().setVirtualFileSystem(
          Result->StatCache->getConsumingFS(VFS));
      // While extending the life of FileMgr and VFS, StatCache should also be
      // extended.
      Ctx->setStatCache(Result->StatCache);

      PreambleCallback(std::move(*Ctx), Result->Pragmas);
    }
    return Result;
  }

  elog("Could not build a preamble for file {0} version {1}: {2}", FileName,
       Inputs.Version, BuiltPreamble.getError().message());
  for (const Diag &D : PreambleDiagnostics.take()) {
    if (D.Severity < DiagnosticsEngine::Error)
      continue;
    // Not an ideal way to show errors, but better than nothing!
    elog("  error: {0}", D.Message);
  }
  return nullptr;
}

bool isPreambleCompatible(const PreambleData &Preamble,
                          const ParseInputs &Inputs, PathRef FileName,
                          const CompilerInvocation &CI) {
  auto ContentsBuffer =
      llvm::MemoryBuffer::getMemBuffer(Inputs.Contents, FileName);
  auto Bounds = ComputePreambleBounds(CI.getLangOpts(), *ContentsBuffer, 0);
  auto VFS = Inputs.TFS->view(Inputs.CompileCommand.Directory);
  return compileCommandsAreEqual(Inputs.CompileCommand,
                                 Preamble.CompileCommand) &&
         Preamble.Preamble.CanReuse(CI, *ContentsBuffer, Bounds, *VFS) &&
         (!Preamble.RequiredModules ||
          Preamble.RequiredModules->canReuse(CI, VFS));
}

void escapeBackslashAndQuotes(llvm::StringRef Text, llvm::raw_ostream &OS) {
  for (char C : Text) {
    switch (C) {
    case '\\':
    case '"':
      OS << '\\';
      break;
    default:
      break;
    }
    OS << C;
  }
}

// Translate diagnostics from baseline into modified for the lines that have the
// same spelling.
static std::vector<Diag> patchDiags(llvm::ArrayRef<Diag> BaselineDiags,
                                    const ScannedPreamble &BaselineScan,
                                    const ScannedPreamble &ModifiedScan) {
  std::vector<Diag> PatchedDiags;
  if (BaselineDiags.empty())
    return PatchedDiags;
  DiagPatcher Patcher(BaselineScan.Lines, ModifiedScan.Lines);
  for (auto &D : BaselineDiags) {
    if (auto NewD = Patcher.translateDiag(D))
      PatchedDiags.emplace_back(std::move(*NewD));
  }
  return PatchedDiags;
}

static std::string getPatchName(llvm::StringRef FileName) {
  // This shouldn't coincide with any real file name.
  llvm::SmallString<128> PatchName;
  llvm::sys::path::append(PatchName, llvm::sys::path::parent_path(FileName),
                          PreamblePatch::HeaderName);
  return PatchName.str().str();
}

PreamblePatch PreamblePatch::create(llvm::StringRef FileName,
                                    const ParseInputs &Modified,
                                    const PreambleData &Baseline,
                                    PatchType PatchType) {
  trace::Span Tracer("CreatePreamblePatch");
  SPAN_ATTACH(Tracer, "File", FileName);
  assert(llvm::sys::path::is_absolute(FileName) && "relative FileName!");
  // First scan preprocessor directives in Baseline and Modified. These will be
  // used to figure out newly added directives in Modified. Scanning can fail,
  // the code just bails out and creates an empty patch in such cases, as:
  // - If scanning for Baseline fails, no knowledge of existing includes hence
  //   patch will contain all the includes in Modified. Leading to rebuild of
  //   whole preamble, which is terribly slow.
  // - If scanning for Modified fails, cannot figure out newly added ones so
  //   there's nothing to do but generate an empty patch.
  auto BaselineScan =
      scanPreamble(Baseline.Preamble.getContents(), Modified.CompileCommand);
  if (!BaselineScan) {
    elog("Failed to scan baseline of {0}: {1}", FileName,
         BaselineScan.takeError());
    return PreamblePatch::unmodified(Baseline);
  }
  auto ModifiedScan = scanPreamble(Modified.Contents, Modified.CompileCommand);
  if (!ModifiedScan) {
    elog("Failed to scan modified contents of {0}: {1}", FileName,
         ModifiedScan.takeError());
    return PreamblePatch::unmodified(Baseline);
  }

  bool IncludesChanged = BaselineScan->Includes != ModifiedScan->Includes;
  bool DirectivesChanged =
      BaselineScan->TextualDirectives != ModifiedScan->TextualDirectives;
  if ((PatchType == PatchType::MacroDirectives || !IncludesChanged) &&
      !DirectivesChanged)
    return PreamblePatch::unmodified(Baseline);

  PreamblePatch PP;
  PP.Baseline = &Baseline;
  PP.PatchFileName = getPatchName(FileName);
  PP.ModifiedBounds = ModifiedScan->Bounds;

  llvm::raw_string_ostream Patch(PP.PatchContents);
  // Set default filename for subsequent #line directives
  Patch << "#line 0 \"";
  // FileName part of a line directive is subject to backslash escaping, which
  // might lead to problems on windows especially.
  escapeBackslashAndQuotes(FileName, Patch);
  Patch << "\"\n";

  if (IncludesChanged && PatchType == PatchType::All) {
    // We are only interested in newly added includes, record the ones in
    // Baseline for exclusion.
    llvm::DenseMap<std::pair<tok::PPKeywordKind, llvm::StringRef>,
                   const Inclusion *>
        ExistingIncludes;
    for (const auto &Inc : Baseline.Includes.MainFileIncludes)
      ExistingIncludes[{Inc.Directive, Inc.Written}] = &Inc;
    // There might be includes coming from disabled regions, record these for
    // exclusion too. note that we don't have resolved paths for those.
    for (const auto &Inc : BaselineScan->Includes)
      ExistingIncludes.try_emplace({Inc.Directive, Inc.Written});
    // Calculate extra includes that needs to be inserted.
    for (auto &Inc : ModifiedScan->Includes) {
      auto It = ExistingIncludes.find({Inc.Directive, Inc.Written});
      // Include already present in the baseline preamble. Set resolved path and
      // put into preamble includes.
      if (It != ExistingIncludes.end()) {
        if (It->second) {
          // If this header is included in an active region of the baseline
          // preamble, preserve it.
          auto &PatchedInc = PP.PreambleIncludes.emplace_back();
          // Copy everything from existing include, apart from the location,
          // when it's coming from baseline preamble.
          PatchedInc = *It->second;
          PatchedInc.HashLine = Inc.HashLine;
          PatchedInc.HashOffset = Inc.HashOffset;
        }
        continue;
      }
      // Include is new in the modified preamble. Inject it into the patch and
      // use #line to set the presumed location to where it is spelled.
      auto LineCol = offsetToClangLineColumn(Modified.Contents, Inc.HashOffset);
      Patch << llvm::formatv("#line {0}\n", LineCol.first);
      Patch << llvm::formatv(
          "#{0} {1}\n", spellingForIncDirective(Inc.Directive), Inc.Written);
    }
  } else {
    // Make sure we have the full set of includes available even when we're not
    // patching. As these are used by features we provide afterwards like hover,
    // go-to-def or include-cleaner when preamble is stale.
    PP.PreambleIncludes = Baseline.Includes.MainFileIncludes;
  }

  if (DirectivesChanged) {
    // We need to patch all the directives, since they are order dependent. e.g:
    // #define BAR(X) NEW(X) // Newly introduced in Modified
    // #define BAR(X) OLD(X) // Exists in the Baseline
    //
    // If we've patched only the first directive, the macro definition would've
    // been wrong for the rest of the file, since patch is applied after the
    // baseline preamble.
    //
    // Note that we deliberately ignore conditional directives and undefs to
    // reduce complexity. The former might cause problems because scanning is
    // imprecise and might pick directives from disabled regions.
    for (const auto &TD : ModifiedScan->TextualDirectives) {
      // Introduce an #undef directive before #defines to suppress any
      // re-definition warnings.
      if (TD.Directive == tok::pp_define)
        Patch << "#undef " << TD.MacroName << '\n';
      Patch << "#line " << TD.DirectiveLine << '\n';
      Patch << TD.Text << '\n';
    }
  }

  PP.PatchedDiags = patchDiags(Baseline.Diags, *BaselineScan, *ModifiedScan);
  PP.PatchedMarks = std::move(ModifiedScan->Marks);
  PP.PatchedMacros = std::move(ModifiedScan->Macros);
  dlog("Created preamble patch: {0}", Patch.str());
  Patch.flush();
  return PP;
}

PreamblePatch PreamblePatch::createFullPatch(llvm::StringRef FileName,
                                             const ParseInputs &Modified,
                                             const PreambleData &Baseline) {
  return create(FileName, Modified, Baseline, PatchType::All);
}

PreamblePatch PreamblePatch::createMacroPatch(llvm::StringRef FileName,
                                              const ParseInputs &Modified,
                                              const PreambleData &Baseline) {
  return create(FileName, Modified, Baseline, PatchType::MacroDirectives);
}

void PreamblePatch::apply(CompilerInvocation &CI) const {
  // Make sure the compilation uses same target opts as the preamble. Clang has
  // no guarantees around using arbitrary options when reusing PCHs, and
  // different target opts can result in crashes, see
  // ParsedASTTest.PreambleWithDifferentTarget.
  // Make sure this is a deep copy, as the same Baseline might be used
  // concurrently.
  *CI.TargetOpts = *Baseline->TargetOpts;

  // No need to map an empty file.
  if (PatchContents.empty())
    return;
  auto &PPOpts = CI.getPreprocessorOpts();
  auto PatchBuffer =
      // we copy here to ensure contents are still valid if CI outlives the
      // PreamblePatch.
      llvm::MemoryBuffer::getMemBufferCopy(PatchContents, PatchFileName);
  // CI will take care of the lifetime of the buffer.
  PPOpts.addRemappedFile(PatchFileName, PatchBuffer.release());
  // The patch will be parsed after loading the preamble ast and before parsing
  // the main file.
  PPOpts.Includes.push_back(PatchFileName);
}

std::vector<Inclusion> PreamblePatch::preambleIncludes() const {
  return PreambleIncludes;
}

PreamblePatch PreamblePatch::unmodified(const PreambleData &Preamble) {
  PreamblePatch PP;
  PP.Baseline = &Preamble;
  PP.PreambleIncludes = Preamble.Includes.MainFileIncludes;
  PP.ModifiedBounds = Preamble.Preamble.getBounds();
  PP.PatchedDiags = Preamble.Diags;
  return PP;
}

llvm::ArrayRef<PragmaMark> PreamblePatch::marks() const {
  if (PatchContents.empty())
    return Baseline->Marks;
  return PatchedMarks;
}

const MainFileMacros &PreamblePatch::mainFileMacros() const {
  if (PatchContents.empty())
    return Baseline->Macros;
  return PatchedMacros;
}

OptionalFileEntryRef PreamblePatch::getPatchEntry(llvm::StringRef MainFilePath,
                                                  const SourceManager &SM) {
  auto PatchFilePath = getPatchName(MainFilePath);
  return SM.getFileManager().getOptionalFileRef(PatchFilePath);
}
} // namespace clangd
} // namespace clang
