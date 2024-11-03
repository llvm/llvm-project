//===--- AnalysisInternal.h - Analysis building blocks ------------- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides smaller, testable pieces of the used-header analysis.
// We find the headers by chaining together several mappings.
//
// AST => AST node => Symbol => Location => Header
//                   /
// Macro expansion =>
//
// The individual steps are declared here.
// (AST => AST Node => Symbol is one API to avoid materializing DynTypedNodes).
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_INCLUDE_CLEANER_ANALYSISINTERNAL_H
#define CLANG_INCLUDE_CLEANER_ANALYSISINTERNAL_H

#include "TypesInternal.h"
#include "clang-include-cleaner/Record.h"
#include "clang-include-cleaner/Types.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <vector>

namespace clang {
class ASTContext;
class Decl;
class HeaderSearch;
class NamedDecl;
class SourceLocation;
namespace include_cleaner {

/// Traverses part of the AST from \p Root, finding uses of symbols.
///
/// Each use is reported to the callback:
/// - the SourceLocation describes where the symbol was used. This is usually
///   the primary location of the AST node found under Root.
/// - the NamedDecl is the symbol referenced. It is canonical, rather than e.g.
///   the redecl actually found by lookup.
/// - the RefType describes the relation between the SourceLocation and the
///   NamedDecl.
///
/// walkAST is typically called once per top-level declaration in the file
/// being analyzed, in order to find all references within it.
void walkAST(Decl &Root,
             llvm::function_ref<void(SourceLocation, NamedDecl &, RefType)>);

/// Finds the headers that provide the symbol location.
llvm::SmallVector<Hinted<Header>> findHeaders(const SymbolLocation &Loc,
                                              const SourceManager &SM,
                                              const PragmaIncludes *PI);

/// A set of locations that provides the declaration.
std::vector<Hinted<SymbolLocation>> locateSymbol(const Symbol &S);

/// Gets all the providers for a symbol by traversing each location.
/// Returned headers are sorted by relevance, first element is the most
/// likely provider for the symbol.
llvm::SmallVector<Header> headersForSymbol(const Symbol &S,
                                           const SourceManager &SM,
                                           const PragmaIncludes *PI);

/// Write an HTML summary of the analysis to the given stream.
void writeHTMLReport(FileID File, const Includes &,
                     llvm::ArrayRef<Decl *> Roots,
                     llvm::ArrayRef<SymbolReference> MacroRefs, ASTContext &Ctx,
                     HeaderSearch &HS, PragmaIncludes *PI,
                     llvm::raw_ostream &OS);

} // namespace include_cleaner
} // namespace clang

#endif
