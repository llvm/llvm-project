//===--- Preamble.h - Reusing expensive parts of the AST ---------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The vast majority of code in a typical translation unit is in the headers
// included at the top of the file.
//
// The preamble optimization says that we can parse this code once, and reuse
// the result multiple times. The preamble is invalidated by changes to the
// code in the preamble region, to the compile command, or to files on disk.
//
// This is the most important optimization in clangd: it allows operations like
// code-completion to have sub-second latency. It is supported by the
// PrecompiledPreamble functionality in clang, which wraps the techniques used
// by PCH files, modules etc into a convenient interface.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_PREAMBLE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_PREAMBLE_H

#include "CollectMacros.h"
#include "Compiler.h"
#include "Diagnostics.h"
#include "FS.h"
#include "Headers.h"
#include "index/CanonicalIncludes.h"
#include "clang/Frontend/PrecompiledPreamble.h"
#include "clang/Tooling/CompilationDatabase.h"

#include <memory>
#include <string>
#include <vector>

namespace clang {
namespace clangd {

/// The parsed preamble and associated data.
///
/// As we must avoid re-parsing the preamble, any information that can only
/// be obtained during parsing must be eagerly captured and stored here.
struct PreambleData {
  PreambleData(PrecompiledPreamble Preamble, std::vector<Diag> Diags,
               IncludeStructure Includes, MainFileMacros Macros,
               std::unique_ptr<PreambleFileStatusCache> StatCache,
               CanonicalIncludes CanonIncludes);

  tooling::CompileCommand CompileCommand;
  PrecompiledPreamble Preamble;
  std::vector<Diag> Diags;
  // Processes like code completions and go-to-definitions will need #include
  // information, and their compile action skips preamble range.
  IncludeStructure Includes;
  // Macros defined in the preamble section of the main file.
  // Users care about headers vs main-file, not preamble vs non-preamble.
  // These should be treated as main-file entities e.g. for code completion.
  MainFileMacros Macros;
  // Cache of FS operations performed when building the preamble.
  // When reusing a preamble, this cache can be consumed to save IO.
  std::unique_ptr<PreambleFileStatusCache> StatCache;
  CanonicalIncludes CanonIncludes;
};

using PreambleParsedCallback =
    std::function<void(ASTContext &, std::shared_ptr<clang::Preprocessor>,
                       const CanonicalIncludes &)>;

/// Build a preamble for the new inputs unless an old one can be reused.
/// If \p OldPreamble can be reused, it is returned unchanged.
/// If \p OldPreamble is null, always builds the preamble.
/// If \p PreambleCallback is set, it will be run on top of the AST while
/// building the preamble. Note that if the old preamble was reused, no AST is
/// built and, therefore, the callback will not be executed.
std::shared_ptr<const PreambleData>
buildPreamble(PathRef FileName, CompilerInvocation &CI,
              std::shared_ptr<const PreambleData> OldPreamble,
              const tooling::CompileCommand &OldCompileCommand,
              const ParseInputs &Inputs, bool StoreInMemory,
              PreambleParsedCallback PreambleCallback);


} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_PREAMBLE_H
