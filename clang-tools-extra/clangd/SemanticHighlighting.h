//==-- SemanticHighlighting.h - Generating highlights from the AST-- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An implementation of semantic highlighting based on this proposal:
// https://github.com/microsoft/vscode-languageserver-node/pull/367 in clangd.
// Semantic highlightings are calculated for an AST by visiting every AST node
// and classifying nodes that are interesting to highlight (variables/function
// calls etc.).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SEMANTICHIGHLIGHTING_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SEMANTICHIGHLIGHTING_H

#include "ClangdUnit.h"
#include "Protocol.h"

namespace clang {
namespace clangd {

enum class HighlightingKind {
  Variable = 0,
  Parameter,
  Function,
  Method,
  Field,
  Class,
  Enum,
  EnumConstant,
  Namespace,
  TemplateParameter,
  Primitive,

  NumKinds,
};

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
};

bool operator==(const LineHighlightings &L, const LineHighlightings &R);

// Returns all HighlightingTokens from an AST. Only generates highlights for the
// main AST.
std::vector<HighlightingToken> getSemanticHighlightings(ParsedAST &AST);

/// Converts a HighlightingKind to a corresponding TextMate scope
/// (https://manual.macromates.com/en/language_grammars).
llvm::StringRef toTextMateScope(HighlightingKind Kind);

/// Convert to LSP's semantic highlighting information.
std::vector<SemanticHighlightingInformation>
toSemanticHighlightingInformation(llvm::ArrayRef<LineHighlightings> Tokens);

/// Return a line-by-line diff between two highlightings.
///  - if the tokens on a line are the same in both hightlightings, this line is
///  omitted.
///  - if a line exists in New but not in Old the tokens on this line are
///  emitted.
///  - if a line does not exists in New but exists in Old an empty line is
///  emitted (to tell client to clear the previous highlightings on this line).
/// \p NewMaxLine is the maximum line number from the new file.
/// REQUIRED: Old and New are sorted.
std::vector<LineHighlightings>
diffHighlightings(ArrayRef<HighlightingToken> New,
                  ArrayRef<HighlightingToken> Old, int NewMaxLine);

} // namespace clangd
} // namespace clang

#endif
