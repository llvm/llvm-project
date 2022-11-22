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

#include "clang-include-cleaner/Record.h"
#include "clang-include-cleaner/Types.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace clang {
class ASTContext;
class Decl;
class NamedDecl;
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

/// A place where a symbol can be provided.
/// It is either a physical file of the TU (SourceLocation) or a logical
/// location in the standard library (stdlib::Symbol).
struct SymbolLocation {
  enum Kind {
    /// A position within a source file (or macro expansion) parsed by clang.
    Physical,
    /// A recognized standard library symbol, like std::string.
    Standard,
  };

  SymbolLocation(SourceLocation S) : Storage(S) {}
  SymbolLocation(tooling::stdlib::Symbol S) : Storage(S) {}

  Kind kind() const { return static_cast<Kind>(Storage.index()); }
  bool operator==(const SymbolLocation &RHS) const {
    return Storage == RHS.Storage;
  }

  SourceLocation physical() const { return std::get<Physical>(Storage); }
  tooling::stdlib::Symbol standard() const {
    return std::get<Standard>(Storage);
  }

private:
  // Order must match Kind enum!
  std::variant<SourceLocation, tooling::stdlib::Symbol> Storage;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Header &);

/// Finds the headers that provide the symbol location.
// FIXME: expose signals
llvm::SmallVector<Header> findHeaders(const SymbolLocation &Loc,
                                      const SourceManager &SM,
                                      const PragmaIncludes &PI);

/// Write an HTML summary of the analysis to the given stream.
/// FIXME: Once analysis has a public API, this should be public too.
void writeHTMLReport(FileID File, llvm::ArrayRef<Decl *> Roots, ASTContext &Ctx,
                     llvm::raw_ostream &OS);

} // namespace include_cleaner
} // namespace clang

#endif
