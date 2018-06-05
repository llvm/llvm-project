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

#include "Headers.h"
#include "Logger.h"
#include "Path.h"
#include "Protocol.h"
#include "index/Index.h"
#include "clang/Frontend/PrecompiledPreamble.h"
#include "clang/Sema/CodeCompleteOptions.h"
#include "clang/Tooling/CompilationDatabase.h"

namespace clang {
class NamedDecl;
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

  /// Add comments to code completion results, if available.
  bool IncludeComments = true;

  /// Include results that are not legal completions in the current context.
  /// For example, private members are usually inaccessible.
  bool IncludeIneligibleResults = false;

  /// Limit the number of results returned (0 means no limit).
  /// If more results are available, we set CompletionList.isIncomplete.
  size_t Limit = 0;

  // Populated internally by clangd, do not set.
  /// If `Index` is set, it is used to augment the code completion
  /// results.
  /// FIXME(ioeric): we might want a better way to pass the index around inside
  /// clangd.
  const SymbolIndex *Index = nullptr;
};

/// Get code completions at a specified \p Pos in \p FileName.
CompletionList codeComplete(PathRef FileName,
                            const tooling::CompileCommand &Command,
                            PrecompiledPreamble const *Preamble,
                            const std::vector<Inclusion> &PreambleInclusions,
                            StringRef Contents, Position Pos,
                            IntrusiveRefCntPtr<vfs::FileSystem> VFS,
                            std::shared_ptr<PCHContainerOperations> PCHs,
                            CodeCompleteOptions Opts);

/// Get signature help at a specified \p Pos in \p FileName.
SignatureHelp signatureHelp(PathRef FileName,
                            const tooling::CompileCommand &Command,
                            PrecompiledPreamble const *Preamble,
                            StringRef Contents, Position Pos,
                            IntrusiveRefCntPtr<vfs::FileSystem> VFS,
                            std::shared_ptr<PCHContainerOperations> PCHs);

// For index-based completion, we only consider:
//   * symbols in namespaces or translation unit scopes (e.g. no class
//     members, no locals)
//   * enum constants in unscoped enum decl (e.g. "red" in "enum {red};")
//   * primary templates (no specializations)
// For the other cases, we let Clang do the completion because it does not
// need any non-local information and it will be much better at following
// lookup rules. Other symbols still appear in the index for other purposes,
// like workspace/symbols or textDocument/definition, but are not used for code
// completion.
bool isIndexedForCodeCompletion(const NamedDecl &ND, ASTContext &ASTCtx);
} // namespace clangd
} // namespace clang

#endif
