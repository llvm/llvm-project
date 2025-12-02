//===--- CreateASTUnitFromArgs.h - Create an ASTUnit from Args ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility for creating an ASTUnit from a vector of command line arguments.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/CreateASTUnitFromArgs.h"
#include "clang/Driver/CreateInvocationFromArgs.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Serialization/ModuleCache.h"
#include "llvm/Support/CrashRecoveryContext.h"

using namespace clang;

/// Create an ASTUnit from a vector of command line arguments, which must
/// specify exactly one source file.
///
/// \param ArgBegin - The beginning of the argument vector.
///
/// \param ArgEnd - The end of the argument vector.
///
/// \param PCHContainerOps - The PCHContainerOperations to use for loading and
/// creating modules.
///
/// \param Diags - The diagnostics engine to use for reporting errors; its
/// lifetime is expected to extend past that of the returned ASTUnit.
///
/// \param ResourceFilesPath - The path to the compiler resource files.
///
/// \param StorePreamblesInMemory - Whether to store PCH in memory. If false,
/// PCH are stored in temporary files.
///
/// \param PreambleStoragePath - The path to a directory, in which to create
/// temporary PCH files. If empty, the default system temporary directory is
/// used. This parameter is ignored if \p StorePreamblesInMemory is true.
///
/// \param ModuleFormat - If provided, uses the specific module format.
///
/// \param ErrAST - If non-null and parsing failed without any AST to return
/// (e.g. because the PCH could not be loaded), this accepts the ASTUnit
/// mainly to allow the caller to see the diagnostics.
///
/// \param VFS - A llvm::vfs::FileSystem to be used for all file accesses.
/// Note that preamble is saved to a temporary directory on a RealFileSystem,
/// so in order for it to be loaded correctly, VFS should have access to
/// it(i.e., be an overlay over RealFileSystem). RealFileSystem will be used
/// if \p VFS is nullptr.
///
// FIXME: Move OnlyLocalDecls, UseBumpAllocator to setters on the ASTUnit, we
// shouldn't need to specify them at construction time.
std::unique_ptr<ASTUnit> clang::CreateASTUnitFromCommandLine(
    const char **ArgBegin, const char **ArgEnd,
    std::shared_ptr<PCHContainerOperations> PCHContainerOps,
    std::shared_ptr<DiagnosticOptions> DiagOpts,
    IntrusiveRefCntPtr<DiagnosticsEngine> Diags, StringRef ResourceFilesPath,
    bool StorePreamblesInMemory, StringRef PreambleStoragePath,
    bool OnlyLocalDecls, CaptureDiagsKind CaptureDiagnostics,
    ArrayRef<ASTUnit::RemappedFile> RemappedFiles,
    bool RemappedFilesKeepOriginalName, unsigned PrecompilePreambleAfterNParses,
    TranslationUnitKind TUKind, bool CacheCodeCompletionResults,
    bool IncludeBriefCommentsInCodeCompletion, bool AllowPCHWithCompilerErrors,
    SkipFunctionBodiesScope SkipFunctionBodies, bool SingleFileParse,
    bool UserFilesAreVolatile, bool ForSerialization,
    bool RetainExcludedConditionalBlocks, std::optional<StringRef> ModuleFormat,
    std::unique_ptr<ASTUnit> *ErrAST,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS) {
  assert(Diags.get() && "no DiagnosticsEngine was provided");

  // If no VFS was provided, create one that tracks the physical file system.
  // If '-working-directory' was passed as an argument, 'createInvocation' will
  // set this as the current working directory of the VFS.
  if (!VFS)
    VFS = llvm::vfs::createPhysicalFileSystem();

  SmallVector<StoredDiagnostic, 4> StoredDiagnostics;

  std::shared_ptr<CompilerInvocation> CI;

  {
    CaptureDroppedDiagnostics Capture(CaptureDiagnostics, *Diags,
                                      &StoredDiagnostics, nullptr);

    CreateInvocationOptions CIOpts;
    CIOpts.VFS = VFS;
    CIOpts.Diags = Diags;
    CIOpts.ProbePrecompiled = true; // FIXME: historical default. Needed?
    CI = createInvocation(llvm::ArrayRef(ArgBegin, ArgEnd), std::move(CIOpts));
    if (!CI)
      return nullptr;
  }

  // Override any files that need remapping
  for (const auto &RemappedFile : RemappedFiles) {
    CI->getPreprocessorOpts().addRemappedFile(RemappedFile.first,
                                              RemappedFile.second);
  }
  PreprocessorOptions &PPOpts = CI->getPreprocessorOpts();
  PPOpts.RemappedFilesKeepOriginalName = RemappedFilesKeepOriginalName;
  PPOpts.AllowPCHWithCompilerErrors = AllowPCHWithCompilerErrors;
  PPOpts.SingleFileParseMode = SingleFileParse;
  PPOpts.RetainExcludedConditionalBlocks = RetainExcludedConditionalBlocks;

  // Override the resources path.
  CI->getHeaderSearchOpts().ResourceDir = std::string(ResourceFilesPath);

  CI->getFrontendOpts().SkipFunctionBodies =
      SkipFunctionBodies == SkipFunctionBodiesScope::PreambleAndMainFile;

  if (ModuleFormat)
    CI->getHeaderSearchOpts().ModuleFormat = std::string(*ModuleFormat);

  // Create the AST unit.
  std::unique_ptr<ASTUnit> AST;
  AST.reset(new ASTUnit(false));
  AST->NumStoredDiagnosticsFromDriver = StoredDiagnostics.size();
  AST->StoredDiagnostics.swap(StoredDiagnostics);
  ASTUnit::ConfigureDiags(Diags, *AST, CaptureDiagnostics);
  AST->DiagOpts = DiagOpts;
  AST->Diagnostics = Diags;
  AST->FileSystemOpts = CI->getFileSystemOpts();
  AST->CodeGenOpts = std::make_unique<CodeGenOptions>(CI->getCodeGenOpts());
  VFS = createVFSFromCompilerInvocation(*CI, *Diags, VFS);
  AST->FileMgr =
      llvm::makeIntrusiveRefCnt<FileManager>(AST->FileSystemOpts, VFS);
  AST->StorePreamblesInMemory = StorePreamblesInMemory;
  AST->PreambleStoragePath = PreambleStoragePath;
  AST->ModCache = createCrossProcessModuleCache();
  AST->OnlyLocalDecls = OnlyLocalDecls;
  AST->CaptureDiagnostics = CaptureDiagnostics;
  AST->TUKind = TUKind;
  AST->ShouldCacheCodeCompletionResults = CacheCodeCompletionResults;
  AST->IncludeBriefCommentsInCodeCompletion =
      IncludeBriefCommentsInCodeCompletion;
  AST->UserFilesAreVolatile = UserFilesAreVolatile;
  AST->Invocation = CI;
  AST->SkipFunctionBodies = SkipFunctionBodies;
  if (ForSerialization)
    AST->WriterData.reset(
        new ASTUnit::ASTWriterData(*AST->ModCache, *AST->CodeGenOpts));
  // Zero out now to ease cleanup during crash recovery.
  CI = nullptr;
  Diags = nullptr;

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<ASTUnit> ASTUnitCleanup(AST.get());

  if (AST->LoadFromCompilerInvocation(std::move(PCHContainerOps),
                                      PrecompilePreambleAfterNParses, VFS)) {
    // Some error occurred, if caller wants to examine diagnostics, pass it the
    // ASTUnit.
    if (ErrAST) {
      AST->StoredDiagnostics.swap(AST->FailedParseDiagnostics);
      ErrAST->swap(AST);
    }
    return nullptr;
  }

  return AST;
}
