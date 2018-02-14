//===--- Compiler.h ---------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// Shared utilities for invoking the clang compiler.
// ClangdUnit takes care of much of this, but some features like CodeComplete
// run their own compile actions that share logic.
//
//===---------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_COMPILER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_COMPILER_H

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/PrecompiledPreamble.h"

namespace clang {
namespace clangd {

class IgnoreDiagnostics : public DiagnosticConsumer {
public:
  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const clang::Diagnostic &Info) override {}
};

/// Creates a CompilerInstance with the main file contens overridden.
/// The preamble will be reused unless it is null.
/// Note that the vfs::FileSystem inside returned instance may differ if
/// additional file remappings occur in command-line arguments.
/// On some errors, returns null. When non-null value is returned, it's expected
/// to be consumed by the FrontendAction as it will have a pointer to the
/// MainFile buffer that will only be deleted if BeginSourceFile is called.
std::unique_ptr<CompilerInstance> prepareCompilerInstance(
    std::unique_ptr<clang::CompilerInvocation>, const PrecompiledPreamble *,
    std::unique_ptr<llvm::MemoryBuffer> MainFile,
    std::shared_ptr<PCHContainerOperations>,
    IntrusiveRefCntPtr<vfs::FileSystem>, DiagnosticConsumer &);

} // namespace clangd
} // namespace clang

#endif
