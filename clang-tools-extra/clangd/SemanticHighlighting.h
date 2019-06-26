//==-- SemanticHighlighting.h - Generating highlights from the AST-- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SEMANTICHIGHLIGHT_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SEMANTICHIGHLIGHT_H

#include "ClangdUnit.h"

namespace clang {
namespace clangd {

enum class HighlightingKind {
  Variable,
  Function,
};

// Contains all information needed for the highlighting a token.
struct HighlightingToken {
  HighlightingKind Kind;
  Range R;
};

bool operator==(const HighlightingToken &Lhs, const HighlightingToken &Rhs);

// Returns all HighlightingTokens from an AST. Only generates highlights for the
// main AST.
std::vector<HighlightingToken> getSemanticHighlightings(ParsedAST &AST);

} // namespace clangd
} // namespace clang

#endif
