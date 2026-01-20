//===--- FindSymbols.h --------------------------------------*- C++-*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Queries that provide a list of symbols matching a string.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_FINDSYMBOLS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_FINDSYMBOLS_H

#include "Protocol.h"
#include "index/Symbol.h"
#include "clang/AST/Decl.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace clangd {
class ParsedAST;
class SymbolIndex;

/// A bitmask type representing symbol tags supported by LSP.
/// \see
/// https://microsoft.github.io/language-server-protocol/specifications/specification-current/#symbolTag
using SymbolTags = uint32_t;
/// Ensure we have enough bits to represent all SymbolTag values.
static_assert(static_cast<unsigned>(SymbolTag::LastTag) <= 32,
              "Too many SymbolTags to fit in uint32_t. Change to uint64_t if "
              "we ever have more than 32 tags.");

/// Helper function for deriving an LSP Location from an index SymbolLocation.
llvm::Expected<Location> indexToLSPLocation(const SymbolLocation &Loc,
                                            llvm::StringRef TUPath);

/// Helper function for deriving an LSP Location for a Symbol.
llvm::Expected<Location> symbolToLocation(const Symbol &Sym,
                                          llvm::StringRef TUPath);

/// Searches for the symbols matching \p Query. The syntax of \p Query can be
/// the non-qualified name or fully qualified of a symbol. For example,
/// "vector" will match the symbol std::vector and "std::vector" would also
/// match it. Direct children of scopes (namespaces, etc) can be listed with a
/// trailing
/// "::". For example, "std::" will list all children of the std namespace and
/// "::" alone will list all children of the global namespace.
/// \p Limit limits the number of results returned (0 means no limit).
/// \p HintPath This is used when resolving URIs. If empty, URI resolution can
/// fail if a hint path is required for the scheme of a specific URI.
llvm::Expected<std::vector<SymbolInformation>>
getWorkspaceSymbols(llvm::StringRef Query, int Limit,
                    const SymbolIndex *const Index, llvm::StringRef HintPath);

/// Retrieves the symbols contained in the "main file" section of an AST in the
/// same order that they appear.
llvm::Expected<std::vector<DocumentSymbol>> getDocumentSymbols(ParsedAST &AST);

/// Converts a single SymbolTag to a bitmask.
SymbolTags toSymbolTagBitmask(SymbolTag ST);

/// Computes symbol tags for a given NamedDecl.
SymbolTags computeSymbolTags(const NamedDecl &ND);

/// Returns the symbol tags for the given declaration.
/// This is a wrapper around computeSymbolTags() which unpacks
/// the tags into a vector.
/// \p ND The declaration to get tags for.
std::vector<SymbolTag> getSymbolTags(const NamedDecl &ND);

} // namespace clangd
} // namespace clang

#endif
