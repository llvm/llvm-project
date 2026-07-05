//===-- CreateInvocationFromArgs.h - Create an ASTUnit from Args-*- C++ -*-===//
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

#ifndef LLVM_CLANG_DRIVER_CREATEASTUNITFROMARGS_H
#define LLVM_CLANG_DRIVER_CREATEASTUNITFROMARGS_H

#include "clang/Frontend/ASTUnit.h"

namespace clang {

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
std::unique_ptr<ASTUnit> CreateASTUnitFromCommandLine(
    const char **ArgBegin, const char **ArgEnd,
    std::shared_ptr<PCHContainerOperations> PCHContainerOps,
    std::shared_ptr<DiagnosticOptions> DiagOpts,
    IntrusiveRefCntPtr<DiagnosticsEngine> Diags, StringRef ResourceFilesPath,
    bool StorePreamblesInMemory = false,
    StringRef PreambleStoragePath = StringRef(), bool OnlyLocalDecls = false,
    CaptureDiagsKind CaptureDiagnostics = CaptureDiagsKind::None,
    ArrayRef<ASTUnit::RemappedFile> RemappedFiles = {},
    bool RemappedFilesKeepOriginalName = true,
    unsigned PrecompilePreambleAfterNParses = 0,
    TranslationUnitKind TUKind = TU_Complete,
    bool CacheCodeCompletionResults = false,
    bool IncludeBriefCommentsInCodeCompletion = false,
    bool AllowPCHWithCompilerErrors = false,
    SkipFunctionBodiesScope SkipFunctionBodies = SkipFunctionBodiesScope::None,
    bool SingleFileParse = false, bool UserFilesAreVolatile = false,
    bool ForSerialization = false, bool RetainExcludedConditionalBlocks = false,
    std::optional<StringRef> ModuleFormat = std::nullopt,
    std::unique_ptr<ASTUnit> *ErrAST = nullptr,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS = nullptr);

} // namespace clang

#endif // LLVM_CLANG_DRIVER_CREATEASTUNITFROMARGS_H
