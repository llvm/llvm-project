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
#include "support/Trace.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
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

/// Gets the includes in the preamble section of the file by running
/// preprocessor over \p Contents. Returned includes do not contain resolved
/// paths. \p VFS and \p Cmd is used to build the compiler invocation, which
/// might stat/read files.
llvm::Expected<std::vector<Inclusion>>
scanPreambleIncludes(llvm::StringRef Contents,
                     llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS,
                     const tooling::CompileCommand &Cmd) {
  // Build and run Preprocessor over the preamble.
  ParseInputs PI;
  PI.Contents = Contents.str();
  PI.FS = std::move(VFS);
  PI.CompileCommand = Cmd;
  IgnoringDiagConsumer IgnoreDiags;
  auto CI = buildCompilerInvocation(PI, IgnoreDiags);
  if (!CI)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to create compiler invocation");
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
      new llvm::vfs::InMemoryFileSystem, IgnoreDiags);
  if (Clang->getFrontendOpts().Inputs.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "compiler instance had no inputs");
  // We are only interested in main file includes.
  Clang->getPreprocessorOpts().SingleFileParseMode = true;
  PreprocessOnlyAction Action;
  if (!Action.BeginSourceFile(*Clang, Clang->getFrontendOpts().Inputs[0]))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed BeginSourceFile");
  Preprocessor &PP = Clang->getPreprocessor();
  IncludeStructure Includes;
  PP.addPPCallbacks(
      collectIncludeStructureCallback(Clang->getSourceManager(), &Includes));
  if (llvm::Error Err = Action.Execute())
    return std::move(Err);
  Action.EndSourceFile();
  return Includes.MainFileIncludes;
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
  if (Inputs.FS->setCurrentWorkingDirectory(Inputs.CompileCommand.Directory)) {
    log("Couldn't set working directory when building the preamble.");
    // We proceed anyway, our lit-tests rely on results for non-existing working
    // dirs.
  }

  llvm::SmallString<32> AbsFileName(FileName);
  Inputs.FS->makeAbsolute(AbsFileName);
  auto StatCache = std::make_unique<PreambleFileStatusCache>(AbsFileName);
  auto BuiltPreamble = PrecompiledPreamble::Build(
      CI, ContentsBuffer.get(), Bounds, *PreambleDiagsEngine,
      StatCache->getProducingFS(Inputs.FS),
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
  return compileCommandsAreEqual(Inputs.CompileCommand,
                                 Preamble.CompileCommand) &&
         Preamble.Preamble.CanReuse(CI, ContentsBuffer.get(), Bounds,
                                    Inputs.FS.get());
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
  assert(llvm::sys::path::is_absolute(FileName) && "relative FileName!");
  // First scan the include directives in Baseline and Modified. These will be
  // used to figure out newly added directives in Modified. Scanning can fail,
  // the code just bails out and creates an empty patch in such cases, as:
  // - If scanning for Baseline fails, no knowledge of existing includes hence
  //   patch will contain all the includes in Modified. Leading to rebuild of
  //   whole preamble, which is terribly slow.
  // - If scanning for Modified fails, cannot figure out newly added ones so
  //   there's nothing to do but generate an empty patch.
  auto BaselineIncludes = scanPreambleIncludes(
      // Contents needs to be null-terminated.
      Baseline.Preamble.getContents().str(),
      Baseline.StatCache->getConsumingFS(Modified.FS), Modified.CompileCommand);
  if (!BaselineIncludes) {
    elog("Failed to scan includes for baseline of {0}: {1}", FileName,
         BaselineIncludes.takeError());
    return {};
  }
  auto ModifiedIncludes = scanPreambleIncludes(
      Modified.Contents, Baseline.StatCache->getConsumingFS(Modified.FS),
      Modified.CompileCommand);
  if (!ModifiedIncludes) {
    elog("Failed to scan includes for modified contents of {0}: {1}", FileName,
         ModifiedIncludes.takeError());
    return {};
  }
  // No patch needed if includes are equal.
  if (*BaselineIncludes == *ModifiedIncludes)
    return PreamblePatch::unmodified(Baseline);

  PreamblePatch PP;
  // This shouldn't coincide with any real file name.
  llvm::SmallString<128> PatchName;
  llvm::sys::path::append(PatchName, llvm::sys::path::parent_path(FileName),
                          "__preamble_patch__.h");
  PP.PatchFileName = PatchName.str().str();

  // We are only interested in newly added includes, record the ones in Baseline
  // for exclusion.
  llvm::DenseMap<std::pair<tok::PPKeywordKind, llvm::StringRef>,
                 /*Resolved=*/llvm::StringRef>
      ExistingIncludes;
  for (const auto &Inc : Baseline.Includes.MainFileIncludes)
    ExistingIncludes[{Inc.Directive, Inc.Written}] = Inc.Resolved;
  // There might be includes coming from disabled regions, record these for
  // exclusion too. note that we don't have resolved paths for those.
  for (const auto &Inc : *BaselineIncludes)
    ExistingIncludes.try_emplace({Inc.Directive, Inc.Written});
  // Calculate extra includes that needs to be inserted.
  llvm::raw_string_ostream Patch(PP.PatchContents);
  // Set default filename for subsequent #line directives
  Patch << "#line 0 \"";
  // FileName part of a line directive is subject to backslash escaping, which
  // might lead to problems on windows especially.
  escapeBackslashAndQuotes(FileName, Patch);
  Patch << "\"\n";
  for (auto &Inc : *ModifiedIncludes) {
    auto It = ExistingIncludes.find({Inc.Directive, Inc.Written});
    // Include already present in the baseline preamble. Set resolved path and
    // put into preamble includes.
    if (It != ExistingIncludes.end()) {
      Inc.Resolved = It->second.str();
      PP.PreambleIncludes.push_back(Inc);
      continue;
    }
    // Include is new in the modified preamble. Inject it into the patch and use
    // #line to set the presumed location to where it is spelled.
    auto LineCol = offsetToClangLineColumn(Modified.Contents, Inc.HashOffset);
    Patch << llvm::formatv("#line {0}\n", LineCol.first);
    Patch << llvm::formatv("#{0} {1}\n", spellingForIncDirective(Inc.Directive),
                           Inc.Written);
  }
  Patch.flush();

  // FIXME: Handle more directives, e.g. define/undef.
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
  return PP;
}

} // namespace clangd
} // namespace clang
