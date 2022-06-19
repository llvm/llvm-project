//===--- Compiler.h ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared utilities for invoking the clang compiler.
// Most callers will use this through Preamble/ParsedAST, but some features like
// CodeComplete run their own compile actions that share these low-level pieces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_COMPILER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_COMPILER_H

#include "FeatureModule.h"
#include "TidyProvider.h"
#include "index/Index.h"
#include "support/ThreadsafeFS.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/PrecompiledPreamble.h"
#include "clang/Tooling/CompilationDatabase.h"
#include <memory>
#include <vector>

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
  bool PreambleParseForwardingFunctions = false;
};

/// Information required to run clang, e.g. to parse AST or do code completion.
struct ParseInputs {
  tooling::CompileCommand CompileCommand;
  const ThreadsafeFS *TFS;
  std::string Contents;
  // Version identifier for Contents, provided by the client and opaque to us.
  std::string Version = "null";
  // Prevent reuse of the cached preamble/AST. Slow! Useful to workaround
  // clangd's assumption that missing header files will stay missing.
  bool ForceRebuild = false;
  // Used to recover from diagnostics (e.g. find missing includes for symbol).
  const SymbolIndex *Index = nullptr;
  ParseOptions Opts = ParseOptions();
  TidyProviderRef ClangTidyProvider = {};
  // Used to acquire ASTListeners when parsing files.
  FeatureModuleSet *FeatureModules = nullptr;
};

/// Clears \p CI from options that are not supported by clangd, like codegen or
/// plugins. This should be combined with CommandMangler::adjust, which provides
/// similar functionality for options that needs to be stripped from compile
/// flags.
void disableUnsupportedOptions(CompilerInvocation &CI);

/// Builds compiler invocation that could be used to build AST or preamble.
std::unique_ptr<CompilerInvocation>
buildCompilerInvocation(const ParseInputs &Inputs, clang::DiagnosticConsumer &D,
                        std::vector<std::string> *CC1Args = nullptr);

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

/// Respect `#pragma clang __debug crash` etc, which are usually disabled.
/// This may only be called before threads are spawned.
void allowCrashPragmasForTest();

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_COMPILER_H
