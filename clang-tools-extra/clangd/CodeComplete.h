//===--- CodeComplete.h ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "Protocol.h"
#include "Quality.h"
#include "index/Index.h"
#include "index/Symbol.h"
#include "index/SymbolOrigin.h"
#include "support/Logger.h"
#include "support/Markup.h"
#include "support/Path.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Sema/CodeCompleteOptions.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <functional>
#include <future>

namespace clang {
class NamedDecl;
namespace clangd {
struct PreambleData;
struct CodeCompletion;

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
  /// If none, the implementation may choose an appropriate behavior.
  /// (In practice, ClangdLSPServer enables bundling if the client claims
  /// to supports signature help).
  llvm::Optional<bool> BundleOverloads;

  /// Limit the number of results returned (0 means no limit).
  /// If more results are available, we set CompletionList.isIncomplete.
  size_t Limit = 0;

  /// Whether to present doc comments as plain-text or markdown.
  MarkupKind DocumentationFormat = MarkupKind::PlainText;

  enum IncludeInsertion {
    IWYU,
    NeverInsert,
  } InsertIncludes = IncludeInsertion::IWYU;

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

  /// Whether to include index symbols that are not defined in the scopes
  /// visible from the code completion point. This applies in contexts without
  /// explicit scope qualifiers.
  ///
  /// Such completions can insert scope qualifiers.
  bool AllScopes = false;

  /// Whether to use the clang parser, or fallback to text-based completion
  /// (using identifiers in the current file and symbol indexes).
  enum CodeCompletionParse {
    /// Block until we can run the parser (e.g. preamble is built).
    /// Return an error if this fails.
    AlwaysParse,
    /// Run the parser if inputs (preamble) are ready.
    /// Otherwise, use text-based completion.
    ParseIfReady,
    /// Always use text-based completion.
    NeverParse,
  } RunParser = ParseIfReady;

  /// Callback invoked on all CompletionCandidate after they are scored and
  /// before they are ranked (by -Score). Thus the results are yielded in
  /// arbitrary order.
  ///
  /// This callbacks allows capturing various internal structures used by clangd
  /// during code completion. Eg: Symbol quality and relevance signals.
  std::function<void(const CodeCompletion &, const SymbolQualitySignals &,
                     const SymbolRelevanceSignals &, float Score)>
      RecordCCResult;
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
  // The parsed documentation comment.
  llvm::Optional<markup::Document> Documentation;
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

  struct IncludeCandidate {
    // The header through which this symbol could be included.
    // Quoted string as expected by an #include directive, e.g. "<memory>".
    // Empty for non-symbol completions, or when not known.
    std::string Header;
    // Present if Header should be inserted to use this item.
    llvm::Optional<TextEdit> Insertion;
  };
  // All possible include headers ranked by preference. By default, the first
  // include is used.
  // If we've bundled together overloads that have different sets of includes,
  // thse includes may not be accurate for all of them.
  llvm::SmallVector<IncludeCandidate, 1> Includes;

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

  /// Indicates if this item is deprecated.
  bool Deprecated = false;

  // Serialize this to an LSP completion item. This is a lossy operation.
  CompletionItem render(const CodeCompleteOptions &) const;
};
raw_ostream &operator<<(raw_ostream &, const CodeCompletion &);
struct CodeCompleteResult {
  std::vector<CodeCompletion> Completions;
  bool HasMore = false;
  CodeCompletionContext::Kind Context = CodeCompletionContext::CCC_Other;
  // The text that is being directly completed.
  // Example: foo.pb^ -> foo.push_back()
  //              ~~
  // Typically matches the textEdit.range of Completions, but not guaranteed to.
  llvm::Optional<Range> CompletionRange;
  // Usually the source will be parsed with a real C++ parser.
  // But heuristics may be used instead if e.g. the preamble is not ready.
  bool RanParser = true;
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

/// Gets code completions at a specified \p Pos in \p FileName.
///
/// If \p Preamble is nullptr, this runs code completion without compiling the
/// code.
///
/// If \p SpecFuzzyFind is set, a speculative and asynchronous fuzzy find index
/// request (based on cached request) will be run before parsing sema. In case
/// the speculative result is used by code completion (e.g. speculation failed),
/// the speculative result is not consumed, and `SpecFuzzyFind` is only
/// destroyed when the async request finishes.
CodeCompleteResult codeComplete(PathRef FileName,
                                const tooling::CompileCommand &Command,
                                const PreambleData *Preamble,
                                StringRef Contents, Position Pos,
                                IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS,
                                CodeCompleteOptions Opts,
                                SpeculativeFuzzyFind *SpecFuzzyFind = nullptr);

/// Get signature help at a specified \p Pos in \p FileName.
SignatureHelp signatureHelp(PathRef FileName,
                            const tooling::CompileCommand &Command,
                            const PreambleData &Preamble, StringRef Contents,
                            Position Pos,
                            IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS,
                            const SymbolIndex *Index);

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

// Text immediately before the completion point that should be completed.
// This is heuristically derived from the source code, and is used when:
//   - semantic analysis fails
//   - semantic analysis may be slow, and we speculatively query the index
struct CompletionPrefix {
  // The unqualified partial name.
  // If there is none, begin() == end() == completion position.
  llvm::StringRef Name;
  // The spelled scope qualifier, such as Foo::.
  // If there is none, begin() == end() == Name.begin().
  llvm::StringRef Qualifier;
};
// Heuristically parses before Offset to determine what should be completed.
CompletionPrefix guessCompletionPrefix(llvm::StringRef Content,
                                       unsigned Offset);

// Whether it makes sense to complete at the point based on typed characters.
// For instance, we implicitly trigger at `a->^` but not at `a>^`.
bool allowImplicitCompletion(llvm::StringRef Content, unsigned Offset);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_CODECOMPLETE_H
