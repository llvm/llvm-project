//===--- Compiler.h ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#include "../clang-tidy/ClangTidyOptions.h"
#include "index/Index.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/PrecompiledPreamble.h"
#include "clang/Tooling/CompilationDatabase.h"

namespace clang {
namespace clangd {

class IgnoreDiagnostics : public DiagnosticConsumer {
public:
  static void log(DiagnosticsEngine::Level DiagLevel,
                  const clang::Diagnostic &Info);

  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const clang::Diagnostic &Info) override;
};

// Options to run clang e.g. when parsing AST.
struct ParseOptions {
  tidy::ClangTidyOptions ClangTidyOpts;
  bool SuggestMissingIncludes = false;
};

/// Information required to run clang, e.g. to parse AST or do code completion.
struct ParseInputs {
  tooling::CompileCommand CompileCommand;
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS;
  std::string Contents;
  // Used to recover from diagnostics (e.g. find missing includes for symbol).
  const SymbolIndex *Index = nullptr;
  ParseOptions Opts;
};

/// Builds compiler invocation that could be used to build AST or preamble.
std::unique_ptr<CompilerInvocation>
buildCompilerInvocation(const ParseInputs &Inputs);

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
    IntrusiveRefCntPtr<llvm::vfs::FileSystem>, DiagnosticConsumer &);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_COMPILER_H
