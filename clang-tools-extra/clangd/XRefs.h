//===--- XRefs.h -------------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Features that traverse references between symbols.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_XREFS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_XREFS_H

#include "ClangdUnit.h"
#include "Protocol.h"
#include "index/Index.h"
#include "llvm/ADT/Optional.h"
#include <vector>

namespace clang {
namespace clangd {

/// Get definition of symbol at a specified \p Pos.
std::vector<Location> findDefinitions(ParsedAST &AST, Position Pos,
                                      const SymbolIndex *Index = nullptr);

/// Returns highlights for all usages of a symbol at \p Pos.
std::vector<DocumentHighlight> findDocumentHighlights(ParsedAST &AST,
                                                      Position Pos);

/// Get the hover information when hovering at \p Pos.
llvm::Optional<Hover> getHover(ParsedAST &AST, Position Pos);

/// Returns reference locations of the symbol at a specified \p Pos.
std::vector<Location> findReferences(ParsedAST &AST, Position Pos,
                                     const SymbolIndex *Index = nullptr);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_XREFS_H
