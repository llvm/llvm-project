//===--- Compiler.h ----------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Shared utilities for invoking the clang compiler.
// ClangdUnit takes care of much of this, but some features like CodeComplete
// run their own compile actions that share logic.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_COMPILER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_COMPILER_H

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/PrecompiledPreamble.h"

namespace clang {
namespace clangd {

class IgnoreDiagnostics : public DiagnosticConsumer {
public:
  static void log(DiagnosticsEngine::Level DiagLevel,
                  const clang::Diagnostic &Info);

  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const clang::Diagnostic &Info) override;
};

/// Creates a compiler instance, configured so that:
///   - Contents of the parsed file are remapped to \p MainFile.
///   - Preamble is overriden to use PCH passed to this function. It means the
///     changes to the preamble headers or files included in the preamble are
///     not visible to this compiler instance.
///   - llvm::vfs::FileSystem is used for all underlying file accesses. The
///     actual vfs used by the compiler may be an overlay over the passed vfs.
/// Returns null on errors. When non-null value is returned, it is expected to
/// be consumed by FrontendAction::BeginSourceFile to properly destroy \p
/// MainFile.
std::unique_ptr<CompilerInstance> prepareCompilerInstance(
    std::unique_ptr<clang::CompilerInvocation>, const PrecompiledPreamble *,
    std::unique_ptr<llvm::MemoryBuffer> MainFile,
    std::shared_ptr<PCHContainerOperations>,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem>, DiagnosticConsumer &);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_COMPILER_H
