//===--- FindSymbols.h --------------------------------------*- C++-*------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Queries that provide a list of symbols matching a string.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_FINDSYMBOLS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_FINDSYMBOLS_H

#include "Protocol.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
class ParsedAST;
namespace clangd {
class SymbolIndex;

/// Searches for the symbols matching \p Query. The syntax of \p Query can be
/// the non-qualified name or fully qualified of a symbol. For example, "vector"
/// will match the symbol std::vector and "std::vector" would also match it.
/// Direct children of scopes (namepaces, etc) can be listed with a trailing
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
llvm::Expected<std::vector<SymbolInformation>>
getDocumentSymbols(ParsedAST &AST);

} // namespace clangd
} // namespace clang

#endif
