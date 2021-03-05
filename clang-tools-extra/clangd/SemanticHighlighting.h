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
// This is particularly valuable for C++: its complex and context-dependent
// grammar is a challenge for simple syntax-highlighting techniques.
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
  Interface,
  Enum,
  EnumConstant,
  Typedef,
  Type,
  Unknown,
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

enum class HighlightingModifier {
  Declaration,
  // FIXME: Definition (needs findExplicitReferences support)
  Deprecated,
  Deduced,
  Readonly,
  Static,
  Abstract,
  DependentName,

  FunctionScope,
  ClassScope,
  FileScope,
  GlobalScope,

  LastModifier = GlobalScope
};
static_assert(static_cast<unsigned>(HighlightingModifier::LastModifier) < 32,
              "Increase width of modifiers bitfield!");
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, HighlightingModifier K);

// Contains all information needed for the highlighting a token.
struct HighlightingToken {
  HighlightingKind Kind;
  uint32_t Modifiers = 0;
  Range R;

  HighlightingToken &addModifier(HighlightingModifier M) {
    Modifiers |= 1 << static_cast<unsigned>(M);
    return *this;
  }
};

bool operator==(const HighlightingToken &L, const HighlightingToken &R);
bool operator<(const HighlightingToken &L, const HighlightingToken &R);

// Returns all HighlightingTokens from an AST. Only generates highlights for the
// main AST.
std::vector<HighlightingToken> getSemanticHighlightings(ParsedAST &AST);

std::vector<SemanticToken> toSemanticTokens(llvm::ArrayRef<HighlightingToken>);
llvm::StringRef toSemanticTokenType(HighlightingKind Kind);
llvm::StringRef toSemanticTokenModifier(HighlightingModifier Modifier);
std::vector<SemanticTokensEdit> diffTokens(llvm::ArrayRef<SemanticToken> Before,
                                           llvm::ArrayRef<SemanticToken> After);

} // namespace clangd
} // namespace clang

#endif
