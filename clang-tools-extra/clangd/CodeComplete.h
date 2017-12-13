//===--- CodeComplete.h -----------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// Code completion provides suggestions for what the user might type next.
// After "std::string S; S." we might suggest members of std::string.
// Signature help describes the parameters of a function as you type them.
//
//===---------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_CODECOMPLETE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_CODECOMPLETE_H

#include "Context.h"
#include "Logger.h"
#include "Path.h"
#include "Protocol.h"
#include "clang/Frontend/PrecompiledPreamble.h"
#include "clang/Sema/CodeCompleteOptions.h"
#include "clang/Tooling/CompilationDatabase.h"

namespace clang {
class PCHContainerOperations;
namespace clangd {

struct CodeCompleteOptions {
  /// Returns options that can be passed to clang's completion engine.
  clang::CodeCompleteOptions getClangCompleteOpts() const;

  /// When true, completion items will contain expandable code snippets in
  /// completion (e.g.  `return ${1:expression}` or `foo(${1:int a}, ${2:int
  /// b})).
  bool EnableSnippets = false;

  /// Add code patterns to completion results.
  /// If EnableSnippets is false, this options is ignored and code patterns will
  /// always be omitted.
  bool IncludeCodePatterns = true;

  /// Add macros to code completion results.
  bool IncludeMacros = true;

  /// Add globals to code completion results.
  bool IncludeGlobals = true;

  /// Add brief comments to completion items, if available.
  /// FIXME(ibiryukov): it looks like turning this option on significantly slows
  /// down completion, investigate if it can be made faster.
  bool IncludeBriefComments = true;

  /// Include results that are not legal completions in the current context.
  /// For example, private members are usually inaccessible.
  bool IncludeIneligibleResults = false;

  /// Limit the number of results returned (0 means no limit).
  /// If more results are available, we set CompletionList.isIncomplete.
  size_t Limit = 0;
};

/// Get code completions at a specified \p Pos in \p FileName.
CompletionList codeComplete(const Context &Ctx, PathRef FileName,
                            const tooling::CompileCommand &Command,
                            PrecompiledPreamble const *Preamble,
                            StringRef Contents, Position Pos,
                            IntrusiveRefCntPtr<vfs::FileSystem> VFS,
                            std::shared_ptr<PCHContainerOperations> PCHs,
                            CodeCompleteOptions Opts);

/// Get signature help at a specified \p Pos in \p FileName.
SignatureHelp signatureHelp(const Context &Ctx, PathRef FileName,
                            const tooling::CompileCommand &Command,
                            PrecompiledPreamble const *Preamble,
                            StringRef Contents, Position Pos,
                            IntrusiveRefCntPtr<vfs::FileSystem> VFS,
                            std::shared_ptr<PCHContainerOperations> PCHs);

} // namespace clangd
} // namespace clang

#endif
