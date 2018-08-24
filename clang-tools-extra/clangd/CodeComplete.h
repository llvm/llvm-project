//===--- CodeComplete.h ------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Code completion provides suggestions for what the user might type next.
// After "std::string S; S." we might suggest members of std::string.
// Signature help describes the parameters of a function as you type them.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_CODECOMPLETE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_CODECOMPLETE_H

#include "Headers.h"
#include "Logger.h"
#include "Path.h"
#include "Protocol.h"
#include "index/Index.h"
#include "clang/Frontend/PrecompiledPreamble.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Sema/CodeCompleteOptions.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <future>

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

  /// Combine overloads into a single completion item where possible.
  bool BundleOverloads = false;

  /// Limit the number of results returned (0 means no limit).
  /// If more results are available, we set CompletionList.isIncomplete.
  size_t Limit = 0;

  /// A visual indicator to prepend to the completion label to indicate whether
  /// completion result would trigger an #include insertion or not.
  struct IncludeInsertionIndicator {
    std::string Insert = "•";
    std::string NoInsert = " ";
  } IncludeIndicator;

  /// Expose origins of completion items in the label (for debugging).
  bool ShowOrigins = false;

  /// If set to true, this will send an asynchronous speculative index request,
  /// based on the index request for the last code completion on the same file
  /// and the filter text typed before the cursor, before sema code completion
  /// is invoked. This can reduce the code completion latency (by roughly
  /// latency of sema code completion) if the speculative request is the same as
  /// the one generated for the ongoing code completion from sema. As a sequence
  /// of code completions often have the same scopes and proximity paths etc,
  /// this should be effective for a number of code completions.
  bool SpeculativeIndexRequest = false;

  // Populated internally by clangd, do not set.
  /// If `Index` is set, it is used to augment the code completion
  /// results.
  /// FIXME(ioeric): we might want a better way to pass the index around inside
  /// clangd.
  const SymbolIndex *Index = nullptr;

  /// Include completions that require small corrections, e.g. change '.' to
  /// '->' on member access etc.
  bool IncludeFixIts = false;

  /// Whether to generate snippets for function arguments on code-completion.
  /// Needs snippets to be enabled as well.
  bool EnableFunctionArgSnippets = true;
};

// Semi-structured representation of a code-complete suggestion for our C++ API.
// We don't use the LSP structures here (unlike most features) as we want
// to expose more data to allow for more precise testing and evaluation.
struct CodeCompletion {
  // The unqualified name of the symbol or other completion item.
  std::string Name;
  // The scope qualifier for the symbol name. e.g. "ns1::ns2::"
  // Empty for non-symbol completions. Not inserted, but may be displayed.
  std::string Scope;
  // Text that must be inserted before the name, and displayed (e.g. base::).
  std::string RequiredQualifier;
  // Details to be displayed following the name. Not inserted.
  std::string Signature;
  // Text to be inserted following the name, in snippet format.
  std::string SnippetSuffix;
  // Type to be displayed for this completion.
  std::string ReturnType;
  std::string Documentation;
  CompletionItemKind Kind = CompletionItemKind::Missing;
  // This completion item may represent several symbols that can be inserted in
  // the same way, such as function overloads. In this case BundleSize > 1, and
  // the following fields are summaries:
  //  - Signature is e.g. "(...)" for functions.
  //  - SnippetSuffix is similarly e.g. "(${0})".
  //  - ReturnType may be empty
  //  - Documentation may be from one symbol, or a combination of several
  // Other fields should apply equally to all bundled completions.
  unsigned BundleSize = 1;
  SymbolOrigin Origin = SymbolOrigin::Unknown;
  // The header through which this symbol could be included.
  // Quoted string as expected by an #include directive, e.g. "<memory>".
  // Empty for non-symbol completions, or when not known.
  std::string Header;
  // Present if Header is set and should be inserted to use this item.
  llvm::Optional<TextEdit> HeaderInsertion;

  /// Holds information about small corrections that needs to be done. Like
  /// converting '->' to '.' on member access.
  std::vector<TextEdit> FixIts;

  /// Holds the range of the token we are going to replace with this completion.
  Range CompletionTokenRange;

  // Scores are used to rank completion items.
  struct Scores {
    // The score that items are ranked by.
    float Total = 0.f;

    // The finalScore with the fuzzy name match score excluded.
    // When filtering client-side, editors should calculate the new fuzzy score,
    // whose scale is 0-1 (with 1 = prefix match, special case 2 = exact match),
    // and recompute finalScore = fuzzyScore * symbolScore.
    float ExcludingName = 0.f;

    // Component scores that contributed to the final score:

    // Quality describes how important we think this candidate is,
    // independent of the query.
    // e.g. symbols with lots of incoming references have higher quality.
    float Quality = 0.f;
    // Relevance describes how well this candidate matched the query.
    // e.g. symbols from nearby files have higher relevance.
    float Relevance = 0.f;
  };
  Scores Score;

  // Serialize this to an LSP completion item. This is a lossy operation.
  CompletionItem render(const CodeCompleteOptions &) const;
};
raw_ostream &operator<<(raw_ostream &, const CodeCompletion &);
struct CodeCompleteResult {
  std::vector<CodeCompletion> Completions;
  bool HasMore = false;
  CodeCompletionContext::Kind Context = CodeCompletionContext::CCC_Other;
};
raw_ostream &operator<<(raw_ostream &, const CodeCompleteResult &);

/// A speculative and asynchronous fuzzy find index request (based on cached
/// request) that can be sent before parsing sema. This would reduce completion
/// latency if the speculation succeeds.
struct SpeculativeFuzzyFind {
  /// A cached request from past code completions.
  /// Set by caller of `codeComplete()`.
  llvm::Optional<FuzzyFindRequest> CachedReq;
  /// The actual request used by `codeComplete()`.
  /// Set by `codeComplete()`. This can be used by callers to update cache.
  llvm::Optional<FuzzyFindRequest> NewReq;
  /// The result is consumed by `codeComplete()` if speculation succeeded.
  /// NOTE: the destructor will wait for the async call to finish.
  std::future<SymbolSlab> Result;
};

/// Get code completions at a specified \p Pos in \p FileName.
/// If \p SpecFuzzyFind is set, a speculative and asynchronous fuzzy find index
/// request (based on cached request) will be run before parsing sema. In case
/// the speculative result is used by code completion (e.g. speculation failed),
/// the speculative result is not consumed, and `SpecFuzzyFind` is only
/// destroyed when the async request finishes.
CodeCompleteResult codeComplete(PathRef FileName,
                                const tooling::CompileCommand &Command,
                                PrecompiledPreamble const *Preamble,
                                const IncludeStructure &PreambleInclusions,
                                StringRef Contents, Position Pos,
                                IntrusiveRefCntPtr<vfs::FileSystem> VFS,
                                std::shared_ptr<PCHContainerOperations> PCHs,
                                CodeCompleteOptions Opts,
                                SpeculativeFuzzyFind *SpecFuzzyFind = nullptr);

/// Get signature help at a specified \p Pos in \p FileName.
SignatureHelp
signatureHelp(PathRef FileName, const tooling::CompileCommand &Command,
              PrecompiledPreamble const *Preamble, StringRef Contents,
              Position Pos, IntrusiveRefCntPtr<vfs::FileSystem> VFS,
              std::shared_ptr<PCHContainerOperations> PCHs, SymbolIndex *Index);

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

/// Retrives a speculative code completion filter text before the cursor.
/// Exposed for testing only.
llvm::Expected<llvm::StringRef>
speculateCompletionFilter(llvm::StringRef Content, Position Pos);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_CODECOMPLETE_H
