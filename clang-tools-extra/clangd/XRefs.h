//===--- XRefs.h -------------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// Describes where a symbol is declared and defined (as far as clangd knows).
// There are three cases:
//  - a declaration only, no definition is known (e.g. only header seen)
//  - a declaration and a distinct definition (e.g. function declared in header)
//  - a declaration and an equal definition (e.g. inline function, or class)
// For some types of symbol, e.g. macros, definition == declaration always.
struct LocatedSymbol {
  // The (unqualified) name of the symbol.
  std::string Name;
  // The canonical or best declaration: where most users find its interface.
  Location PreferredDeclaration;
  // Where the symbol is defined, if known. May equal PreferredDeclaration.
  llvm::Optional<Location> Definition;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const LocatedSymbol &);
/// Get definition of symbol at a specified \p Pos.
/// Multiple locations may be returned, corresponding to distinct symbols.
std::vector<LocatedSymbol> locateSymbolAt(ParsedAST &AST, Position Pos,
                                          const SymbolIndex *Index = nullptr);

/// Returns highlights for all usages of a symbol at \p Pos.
std::vector<DocumentHighlight> findDocumentHighlights(ParsedAST &AST,
                                                      Position Pos);

/// Get the hover information when hovering at \p Pos.
llvm::Optional<Hover> getHover(ParsedAST &AST, Position Pos);

/// Returns reference locations of the symbol at a specified \p Pos.
/// \p Limit limits the number of results returned (0 means no limit).
std::vector<Location> findReferences(ParsedAST &AST, Position Pos,
                                     uint32_t Limit,
                                     const SymbolIndex *Index = nullptr);

/// Get info about symbols at \p Pos.
std::vector<SymbolDetails> getSymbolInfo(ParsedAST &AST, Position Pos);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_XREFS_H
