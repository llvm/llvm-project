//===--- Compiler.cpp --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Compiler.h"
#include "support/Logger.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Serialization/PCHContainerOperations.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace clangd {

void IgnoreDiagnostics::log(DiagnosticsEngine::Level DiagLevel,
                            const clang::Diagnostic &Info) {
  // FIXME: format lazily, in case vlog is off.
  llvm::SmallString<64> Message;
  Info.FormatDiagnostic(Message);

  llvm::SmallString<64> Location;
  if (Info.hasSourceManager() && Info.getLocation().isValid()) {
    auto &SourceMgr = Info.getSourceManager();
    auto Loc = SourceMgr.getFileLoc(Info.getLocation());
    llvm::raw_svector_ostream OS(Location);
    Loc.print(OS, SourceMgr);
    OS << ":";
  }

  clangd::vlog("Ignored diagnostic. {0}{1}", Location, Message);
}

void IgnoreDiagnostics::HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                                         const clang::Diagnostic &Info) {
  IgnoreDiagnostics::log(DiagLevel, Info);
}

static bool AllowCrashPragmasForTest = false;
void allowCrashPragmasForTest() { AllowCrashPragmasForTest = true; }

void disableUnsupportedOptions(CompilerInvocation &CI) {
  // Disable "clang -verify" diagnostics, they are rarely useful in clangd, and
  // our compiler invocation set-up doesn't seem to work with it (leading
  // assertions in VerifyDiagnosticConsumer).
  CI.getDiagnosticOpts().VerifyDiagnostics = false;
  CI.getDiagnosticOpts().ShowColors = false;

  // Disable any dependency outputting, we don't want to generate files or write
  // to stdout/stderr.
  CI.getDependencyOutputOpts().ShowIncludesDest = ShowIncludesDestination::None;
  CI.getDependencyOutputOpts().OutputFile.clear();
  CI.getDependencyOutputOpts().HeaderIncludeOutputFile.clear();
  CI.getDependencyOutputOpts().DOTOutputFile.clear();
  CI.getDependencyOutputOpts().ModuleDependencyOutputDir.clear();

  // Disable any pch generation/usage operations. Since serialized preamble
  // format is unstable, using an incompatible one might result in unexpected
  // behaviours, including crashes.
  CI.getPreprocessorOpts().ImplicitPCHInclude.clear();
  CI.getPreprocessorOpts().PrecompiledPreambleBytes = {0, false};
  CI.getPreprocessorOpts().PCHThroughHeader.clear();
  CI.getPreprocessorOpts().PCHWithHdrStop = false;
  CI.getPreprocessorOpts().PCHWithHdrStopCreate = false;
  // Don't crash on `#pragma clang __debug parser_crash`
  if (!AllowCrashPragmasForTest)
    CI.getPreprocessorOpts().DisablePragmaDebugCrash = true;

  // Always default to raw container format as clangd doesn't registry any other
  // and clang dies when faced with unknown formats.
  CI.getHeaderSearchOpts().ModuleFormat =
      PCHContainerOperations().getRawReader().getFormat().str();

  CI.getFrontendOpts().Plugins.clear();
  CI.getFrontendOpts().AddPluginActions.clear();
  CI.getFrontendOpts().PluginArgs.clear();
  CI.getFrontendOpts().ProgramAction = frontend::ParseSyntaxOnly;
  CI.getFrontendOpts().ActionName.clear();
}

std::unique_ptr<CompilerInvocation>
buildCompilerInvocation(const ParseInputs &Inputs, clang::DiagnosticConsumer &D,
                        std::vector<std::string> *CC1Args) {
  if (Inputs.CompileCommand.CommandLine.empty())
    return nullptr;
  std::vector<const char *> ArgStrs;
  for (const auto &S : Inputs.CompileCommand.CommandLine)
    ArgStrs.push_back(S.c_str());

  CreateInvocationOptions CIOpts;
  CIOpts.VFS = Inputs.TFS->view(Inputs.CompileCommand.Directory);
  CIOpts.CC1Args = CC1Args;
  CIOpts.RecoverOnError = true;
  CIOpts.Diags =
      CompilerInstance::createDiagnostics(new DiagnosticOptions, &D, false);
  std::unique_ptr<CompilerInvocation> CI = createInvocation(ArgStrs, CIOpts);
  if (!CI)
    return nullptr;
  // createInvocationFromCommandLine sets DisableFree.
  CI->getFrontendOpts().DisableFree = false;
  CI->getLangOpts()->CommentOpts.ParseAllComments = true;
  CI->getLangOpts()->RetainCommentsFromSystemHeaders = true;

  disableUnsupportedOptions(*CI);
  return CI;
}

std::unique_ptr<CompilerInstance>
prepareCompilerInstance(std::unique_ptr<clang::CompilerInvocation> CI,
                        const PrecompiledPreamble *Preamble,
                        std::unique_ptr<llvm::MemoryBuffer> Buffer,
                        llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS,
                        DiagnosticConsumer &DiagsClient) {
  assert(VFS && "VFS is null");
  assert(!CI->getPreprocessorOpts().RetainRemappedFileBuffers &&
         "Setting RetainRemappedFileBuffers to true will cause a memory leak "
         "of ContentsBuffer");

  // NOTE: we use Buffer.get() when adding remapped files, so we have to make
  // sure it will be released if no error is emitted.
  if (Preamble) {
    Preamble->OverridePreamble(*CI, VFS, Buffer.get());
  } else {
    CI->getPreprocessorOpts().addRemappedFile(
        CI->getFrontendOpts().Inputs[0].getFile(), Buffer.get());
  }

  auto Clang = std::make_unique<CompilerInstance>(
      std::make_shared<PCHContainerOperations>());
  Clang->setInvocation(std::move(CI));
  Clang->createDiagnostics(&DiagsClient, false);

  if (auto VFSWithRemapping = createVFSFromCompilerInvocation(
          Clang->getInvocation(), Clang->getDiagnostics(), VFS))
    VFS = VFSWithRemapping;
  Clang->createFileManager(VFS);

  if (!Clang->createTarget())
    return nullptr;

  // RemappedFileBuffers will handle the lifetime of the Buffer pointer,
  // release it.
  Buffer.release();
  return Clang;
}

} // namespace clangd
} // namespace clang
