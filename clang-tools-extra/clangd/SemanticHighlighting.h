//==-- SemanticHighlighting.h - Generating highlights from the AST-- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file supports semantic highlighting: categorizing tokens in the file so
// that the editor can color/style them differently.
//
// This is particularly valuable for C++: its complex and context-dependent
// grammar is a challenge for simple syntax-highlighting techniques.
//
// We support two protocols for providing highlights to the client:
// - the `textDocument/semanticTokens` request from LSP 3.16
//   https://github.com/microsoft/vscode-languageserver-node/blob/release/protocol/3.16.0-next.1/protocol/src/protocol.semanticTokens.proposed.ts
// - the earlier proposed `textDocument/semanticHighlighting` notification
//   https://github.com/microsoft/vscode-languageserver-node/pull/367
//   This is referred to as "Theia" semantic highlighting in the code.
//   It was supported from clangd 9 but should be considered deprecated as of
//   clangd 11 and eventually removed.
//
// Semantic highlightings are calculated for an AST by visiting every AST node
// and classifying nodes that are interesting to highlight (variables/function
// calls etc.).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SEMANTICHIGHLIGHTING_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SEMANTICHIGHLIGHTING_H

#include "Protocol.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace clangd {
class ParsedAST;

enum class HighlightingKind {
  Variable = 0,
  LocalVariable,
  Parameter,
  Function,
  Method,
  StaticMethod,
  Field,
  StaticField,
  Class,
  Enum,
  EnumConstant,
  Typedef,
  DependentType,
  DependentName,
  Namespace,
  TemplateParameter,
  Concept,
  Primitive,
  Macro,

  // This one is different from the other kinds as it's a line style
  // rather than a token style.
  InactiveCode,

  LastKind = InactiveCode
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, HighlightingKind K);

// Contains all information needed for the highlighting a token.
struct HighlightingToken {
  HighlightingKind Kind;
  Range R;
};

bool operator==(const HighlightingToken &L, const HighlightingToken &R);
bool operator<(const HighlightingToken &L, const HighlightingToken &R);

/// Contains all information about highlightings on a single line.
struct LineHighlightings {
  int Line;
  std::vector<HighlightingToken> Tokens;
  bool IsInactive;
};

bool operator==(const LineHighlightings &L, const LineHighlightings &R);

// Returns all HighlightingTokens from an AST. Only generates highlights for the
// main AST.
std::vector<HighlightingToken> getSemanticHighlightings(ParsedAST &AST);

std::vector<SemanticToken> toSemanticTokens(llvm::ArrayRef<HighlightingToken>);
llvm::StringRef toSemanticTokenType(HighlightingKind Kind);
std::vector<SemanticTokensEdit> diffTokens(llvm::ArrayRef<SemanticToken> Before,
                                           llvm::ArrayRef<SemanticToken> After);

/// Converts a HighlightingKind to a corresponding TextMate scope
/// (https://manual.macromates.com/en/language_grammars).
llvm::StringRef toTextMateScope(HighlightingKind Kind);

/// Convert to LSP's semantic highlighting information.
std::vector<TheiaSemanticHighlightingInformation>
toTheiaSemanticHighlightingInformation(
    llvm::ArrayRef<LineHighlightings> Tokens);

/// Return a line-by-line diff between two highlightings.
///  - if the tokens on a line are the same in both highlightings, this line is
///  omitted.
///  - if a line exists in New but not in Old, the tokens on this line are
///  emitted.
///  - if a line does not exist in New but exists in Old, an empty line is
///  emitted (to tell client to clear the previous highlightings on this line).
///
/// REQUIRED: Old and New are sorted.
std::vector<LineHighlightings>
diffHighlightings(ArrayRef<HighlightingToken> New,
                  ArrayRef<HighlightingToken> Old);

} // namespace clangd
} // namespace clang

#endif
