//===--- Preamble.cpp - Reusing expensive parts of the AST ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Preamble.h"
#include "Compiler.h"
#include "Headers.h"
#include "SourceCode.h"
#include "support/Logger.h"
#include "support/ThreadsafeFS.h"
#include "support/Trace.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <iterator>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace clang {
namespace clangd {
namespace {
constexpr llvm::StringLiteral PreamblePatchHeaderName = "__preamble_patch__.h";

bool compileCommandsAreEqual(const tooling::CompileCommand &LHS,
                             const tooling::CompileCommand &RHS) {
  // We don't check for Output, it should not matter to clangd.
  return LHS.Directory == RHS.Directory && LHS.Filename == RHS.Filename &&
         llvm::makeArrayRef(LHS.CommandLine).equals(RHS.CommandLine);
}

class CppFilePreambleCallbacks : public PreambleCallbacks {
public:
  CppFilePreambleCallbacks(PathRef File, PreambleParsedCallback ParsedCallback)
      : File(File), ParsedCallback(ParsedCallback) {}

  IncludeStructure takeIncludes() { return std::move(Includes); }

  MainFileMacros takeMacros() { return std::move(Macros); }

  CanonicalIncludes takeCanonicalIncludes() { return std::move(CanonIncludes); }

  void AfterExecute(CompilerInstance &CI) override {
    if (!ParsedCallback)
      return;
    trace::Span Tracer("Running PreambleCallback");
    ParsedCallback(CI.getASTContext(), CI.getPreprocessorPtr(), CanonIncludes);
  }

  void BeforeExecute(CompilerInstance &CI) override {
    CanonIncludes.addSystemHeadersMapping(CI.getLangOpts());
    LangOpts = &CI.getLangOpts();
    SourceMgr = &CI.getSourceManager();
  }

  std::unique_ptr<PPCallbacks> createPPCallbacks() override {
    assert(SourceMgr && LangOpts &&
           "SourceMgr and LangOpts must be set at this point");

    return std::make_unique<PPChainedCallbacks>(
        collectIncludeStructureCallback(*SourceMgr, &Includes),
        std::make_unique<CollectMainFileMacros>(*SourceMgr, Macros));
  }

  CommentHandler *getCommentHandler() override {
    IWYUHandler = collectIWYUHeaderMaps(&CanonIncludes);
    return IWYUHandler.get();
  }

  bool shouldSkipFunctionBody(Decl *D) override {
    // Generally we skip function bodies in preambles for speed.
    // We can make exceptions for functions that are cheap to parse and
    // instantiate, widely used, and valuable (e.g. commonly produce errors).
    if (const auto *FT = llvm::dyn_cast<clang::FunctionTemplateDecl>(D)) {
      if (const auto *II = FT->getDeclName().getAsIdentifierInfo())
        // std::make_unique is trivial, and we diagnose bad constructor calls.
        if (II->isStr("make_unique") && FT->isInStdNamespace())
          return false;
    }
    return true;
  }

private:
  PathRef File;
  PreambleParsedCallback ParsedCallback;
  IncludeStructure Includes;
  CanonicalIncludes CanonIncludes;
  MainFileMacros Macros;
  std::unique_ptr<CommentHandler> IWYUHandler = nullptr;
  const clang::LangOptions *LangOpts = nullptr;
  const SourceManager *SourceMgr = nullptr;
};

// Represents directives other than includes, where basic textual information is
// enough.
struct TextualPPDirective {
  unsigned DirectiveLine;
  // Full text that's representing the directive, including the `#`.
  std::string Text;

  bool operator==(const TextualPPDirective &RHS) const {
    return std::tie(DirectiveLine, Text) ==
           std::tie(RHS.DirectiveLine, RHS.Text);
  }
};

// Formats a PP directive consisting of Prefix (e.g. "#define ") and Body ("X
// 10"). The formatting is copied so that the tokens in Body have PresumedLocs
// with correct columns and lines.
std::string spellDirective(llvm::StringRef Prefix,
                           CharSourceRange DirectiveRange,
                           const LangOptions &LangOpts, const SourceManager &SM,
                           unsigned &DirectiveLine) {
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

    const auto *MI = MD->getMacroInfo();
    TD.Text =
        spellDirective("#define ",
                       CharSourceRange::getTokenRange(
                           MI->getDefinitionLoc(), MI->getDefinitionEndLoc()),
                       LangOpts, SM, TD.DirectiveLine);
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
  PreambleBounds Bounds = {0, false};
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
  PI.Contents = Contents.str();
  PI.TFS = &FS;
  PI.CompileCommand = Cmd;
  IgnoringDiagConsumer IgnoreDiags;
  auto CI = buildCompilerInvocation(PI, IgnoreDiags);
  if (!CI)
    return error("failed to create compiler invocation");
  CI->getDiagnosticOpts().IgnoreWarnings = true;
  auto ContentsBuffer = llvm::MemoryBuffer::getMemBuffer(Contents);
  // This means we're scanning (though not preprocessing) the preamble section
  // twice. However, it's important to precisely follow the preamble bounds used
  // elsewhere.
  auto Bounds =
      ComputePreambleBounds(*CI->getLangOpts(), ContentsBuffer.get(), 0);
  auto PreambleContents =
      llvm::MemoryBuffer::getMemBufferCopy(Contents.substr(0, Bounds.Size));
  auto Clang = prepareCompilerInstance(
      std::move(CI), nullptr, std::move(PreambleContents),
      // Provide an empty FS to prevent preprocessor from performing IO. This
      // also implies missing resolved paths for includes.
      FS.view(llvm::None), IgnoreDiags);
  if (Clang->getFrontendOpts().Inputs.empty())
    return error("compiler instance had no inputs");
  // We are only interested in main file includes.
  Clang->getPreprocessorOpts().SingleFileParseMode = true;
  PreprocessOnlyAction Action;
  if (!Action.BeginSourceFile(*Clang, Clang->getFrontendOpts().Inputs[0]))
    return error("failed BeginSourceFile");
  const auto &SM = Clang->getSourceManager();
  Preprocessor &PP = Clang->getPreprocessor();
  IncludeStructure Includes;
  PP.addPPCallbacks(collectIncludeStructureCallback(SM, &Includes));
  ScannedPreamble SP;
  SP.Bounds = Bounds;
  PP.addPPCallbacks(
      std::make_unique<DirectiveCollector>(PP, SP.TextualDirectives));
  if (llvm::Error Err = Action.Execute())
    return std::move(Err);
  Action.EndSourceFile();
  SP.Includes = std::move(Includes.MainFileIncludes);
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

// Checks whether \p FileName is a valid spelling of main file.
bool isMainFile(llvm::StringRef FileName, const SourceManager &SM) {
  auto FE = SM.getFileManager().getFile(FileName);
  return FE && *FE == SM.getFileEntryForID(SM.getMainFileID());
}

} // namespace

PreambleData::PreambleData(const ParseInputs &Inputs,
                           PrecompiledPreamble Preamble,
                           std::vector<Diag> Diags, IncludeStructure Includes,
                           MainFileMacros Macros,
                           std::unique_ptr<PreambleFileStatusCache> StatCache,
                           CanonicalIncludes CanonIncludes)
    : Version(Inputs.Version), CompileCommand(Inputs.CompileCommand),
      Preamble(std::move(Preamble)), Diags(std::move(Diags)),
      Includes(std::move(Includes)), Macros(std::move(Macros)),
      StatCache(std::move(StatCache)), CanonIncludes(std::move(CanonIncludes)) {
}

std::shared_ptr<const PreambleData>
buildPreamble(PathRef FileName, CompilerInvocation CI,
              const ParseInputs &Inputs, bool StoreInMemory,
              PreambleParsedCallback PreambleCallback) {
  // Note that we don't need to copy the input contents, preamble can live
  // without those.
  auto ContentsBuffer =
      llvm::MemoryBuffer::getMemBuffer(Inputs.Contents, FileName);
  auto Bounds =
      ComputePreambleBounds(*CI.getLangOpts(), ContentsBuffer.get(), 0);

  trace::Span Tracer("BuildPreamble");
  SPAN_ATTACH(Tracer, "File", FileName);
  StoreDiags PreambleDiagnostics;
  llvm::IntrusiveRefCntPtr<DiagnosticsEngine> PreambleDiagsEngine =
      CompilerInstance::createDiagnostics(&CI.getDiagnosticOpts(),
                                          &PreambleDiagnostics, false);

  // Skip function bodies when building the preamble to speed up building
  // the preamble and make it smaller.
  assert(!CI.getFrontendOpts().SkipFunctionBodies);
  CI.getFrontendOpts().SkipFunctionBodies = true;
  // We don't want to write comment locations into PCH. They are racy and slow
  // to read back. We rely on dynamic index for the comments instead.
  CI.getPreprocessorOpts().WriteCommentListToPCH = false;

  CppFilePreambleCallbacks SerializedDeclsCollector(FileName, PreambleCallback);
  auto VFS = Inputs.TFS->view(Inputs.CompileCommand.Directory);
  llvm::SmallString<32> AbsFileName(FileName);
  VFS->makeAbsolute(AbsFileName);
  auto StatCache = std::make_unique<PreambleFileStatusCache>(AbsFileName);
  auto BuiltPreamble = PrecompiledPreamble::Build(
      CI, ContentsBuffer.get(), Bounds, *PreambleDiagsEngine,
      StatCache->getProducingFS(VFS),
      std::make_shared<PCHContainerOperations>(), StoreInMemory,
      SerializedDeclsCollector);

  // When building the AST for the main file, we do want the function
  // bodies.
  CI.getFrontendOpts().SkipFunctionBodies = false;

  if (BuiltPreamble) {
    vlog("Built preamble of size {0} for file {1} version {2}",
         BuiltPreamble->getSize(), FileName, Inputs.Version);
    std::vector<Diag> Diags = PreambleDiagnostics.take();
    return std::make_shared<PreambleData>(
        Inputs, std::move(*BuiltPreamble), std::move(Diags),
        SerializedDeclsCollector.takeIncludes(),
        SerializedDeclsCollector.takeMacros(), std::move(StatCache),
        SerializedDeclsCollector.takeCanonicalIncludes());
  } else {
    elog("Could not build a preamble for file {0} version {1}", FileName,
         Inputs.Version);
    return nullptr;
  }
}

bool isPreambleCompatible(const PreambleData &Preamble,
                          const ParseInputs &Inputs, PathRef FileName,
                          const CompilerInvocation &CI) {
  auto ContentsBuffer =
      llvm::MemoryBuffer::getMemBuffer(Inputs.Contents, FileName);
  auto Bounds =
      ComputePreambleBounds(*CI.getLangOpts(), ContentsBuffer.get(), 0);
  auto VFS = Inputs.TFS->view(Inputs.CompileCommand.Directory);
  return compileCommandsAreEqual(Inputs.CompileCommand,
                                 Preamble.CompileCommand) &&
         Preamble.Preamble.CanReuse(CI, ContentsBuffer.get(), Bounds,
                                    VFS.get());
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

PreamblePatch PreamblePatch::create(llvm::StringRef FileName,
                                    const ParseInputs &Modified,
                                    const PreambleData &Baseline) {
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
  auto BaselineScan = scanPreamble(
      // Contents needs to be null-terminated.
      Baseline.Preamble.getContents().str(), Modified.CompileCommand);
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
  if (!IncludesChanged && !DirectivesChanged)
    return PreamblePatch::unmodified(Baseline);

  PreamblePatch PP;
  // This shouldn't coincide with any real file name.
  llvm::SmallString<128> PatchName;
  llvm::sys::path::append(PatchName, llvm::sys::path::parent_path(FileName),
                          PreamblePatchHeaderName);
  PP.PatchFileName = PatchName.str().str();
  PP.ModifiedBounds = ModifiedScan->Bounds;

  llvm::raw_string_ostream Patch(PP.PatchContents);
  // Set default filename for subsequent #line directives
  Patch << "#line 0 \"";
  // FileName part of a line directive is subject to backslash escaping, which
  // might lead to problems on windows especially.
  escapeBackslashAndQuotes(FileName, Patch);
  Patch << "\"\n";

  if (IncludesChanged) {
    // We are only interested in newly added includes, record the ones in
    // Baseline for exclusion.
    llvm::DenseMap<std::pair<tok::PPKeywordKind, llvm::StringRef>,
                   /*Resolved=*/llvm::StringRef>
        ExistingIncludes;
    for (const auto &Inc : Baseline.Includes.MainFileIncludes)
      ExistingIncludes[{Inc.Directive, Inc.Written}] = Inc.Resolved;
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
        Inc.Resolved = It->second.str();
        PP.PreambleIncludes.push_back(Inc);
        continue;
      }
      // Include is new in the modified preamble. Inject it into the patch and
      // use #line to set the presumed location to where it is spelled.
      auto LineCol = offsetToClangLineColumn(Modified.Contents, Inc.HashOffset);
      Patch << llvm::formatv("#line {0}\n", LineCol.first);
      Patch << llvm::formatv(
          "#{0} {1}\n", spellingForIncDirective(Inc.Directive), Inc.Written);
    }
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
      Patch << "#line " << TD.DirectiveLine << '\n';
      Patch << TD.Text << '\n';
    }
  }
  dlog("Created preamble patch: {0}", Patch.str());
  Patch.flush();
  return PP;
}

void PreamblePatch::apply(CompilerInvocation &CI) const {
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
  PP.PreambleIncludes = Preamble.Includes.MainFileIncludes;
  PP.ModifiedBounds = Preamble.Preamble.getBounds();
  return PP;
}

SourceLocation translatePreamblePatchLocation(SourceLocation Loc,
                                              const SourceManager &SM) {
  auto DefFile = SM.getFileID(Loc);
  if (auto *FE = SM.getFileEntryForID(DefFile)) {
    auto IncludeLoc = SM.getIncludeLoc(DefFile);
    // Preamble patch is included inside the builtin file.
    if (IncludeLoc.isValid() && SM.isWrittenInBuiltinFile(IncludeLoc) &&
        FE->getName().endswith(PreamblePatchHeaderName)) {
      auto Presumed = SM.getPresumedLoc(Loc);
      // Check that line directive is pointing at main file.
      if (Presumed.isValid() && Presumed.getFileID().isInvalid() &&
          isMainFile(Presumed.getFilename(), SM)) {
        Loc = SM.translateLineCol(SM.getMainFileID(), Presumed.getLine(),
                                  Presumed.getColumn());
      }
    }
  }
  return Loc;
}
} // namespace clangd
} // namespace clang
