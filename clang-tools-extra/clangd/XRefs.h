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
#include "FormattedString.h"
#include "Protocol.h"
#include "index/Index.h"
#include "index/SymbolLocation.h"
#include "clang/Index/IndexSymbol.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/raw_ostream.h"
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

/// Contains detailed information about a Symbol. Especially useful when
/// generating hover responses. It can be rendered as a hover panel, or
/// embedding clients can use the structured information to provide their own
/// UI.
struct HoverInfo {
  /// Represents parameters of a function, a template or a macro.
  /// For example:
  /// - void foo(ParamType Name = DefaultValue)
  /// - #define FOO(Name)
  /// - template <ParamType Name = DefaultType> class Foo {};
  struct Param {
    /// The pretty-printed parameter type, e.g. "int", or "typename" (in
    /// TemplateParameters)
    llvm::Optional<std::string> Type;
    /// None for unnamed parameters.
    llvm::Optional<std::string> Name;
    /// None if no default is provided.
    llvm::Optional<std::string> Default;
  };

  /// For a variable named Bar, declared in clang::clangd::Foo::getFoo the
  /// following fields will hold:
  /// - NamespaceScope: clang::clangd::
  /// - LocalScope: Foo::getFoo::
  /// - Name: Bar

  /// Scopes might be None in cases where they don't make sense, e.g. macros and
  /// auto/decltype.
  /// Contains all of the enclosing namespaces, empty string means global
  /// namespace.
  llvm::Optional<std::string> NamespaceScope;
  /// Remaining named contexts in symbol's qualified name, empty string means
  /// symbol is not local.
  std::string LocalScope;
  /// Name of the symbol, does not contain any "::".
  std::string Name;
  llvm::Optional<Range> SymRange;
  /// Scope containing the symbol. e.g, "global namespace", "function x::Y"
  /// - None for deduced types, e.g "auto", "decltype" keywords.
  SymbolKind Kind;
  std::string Documentation;
  /// Source code containing the definition of the symbol.
  std::string Definition;

  /// Pretty-printed variable type.
  /// Set only for variables.
  llvm::Optional<std::string> Type;
  /// Set for functions and lambadas.
  llvm::Optional<std::string> ReturnType;
  /// Set for functions, lambdas and macros with parameters.
  llvm::Optional<std::vector<Param>> Parameters;
  /// Set for all templates(function, class, variable).
  llvm::Optional<std::vector<Param>> TemplateParameters;

  /// Produce a user-readable information.
  FormattedString present() const;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const HoverInfo::Param &);
inline bool operator==(const HoverInfo::Param &LHS,
                       const HoverInfo::Param &RHS) {
  return std::tie(LHS.Type, LHS.Name, LHS.Default) ==
         std::tie(RHS.Type, RHS.Name, RHS.Default);
}

/// Get the hover information when hovering at \p Pos.
llvm::Optional<HoverInfo> getHover(ParsedAST &AST, Position Pos,
                                   format::FormatStyle Style);

/// Returns reference locations of the symbol at a specified \p Pos.
/// \p Limit limits the number of results returned (0 means no limit).
std::vector<Location> findReferences(ParsedAST &AST, Position Pos,
                                     uint32_t Limit,
                                     const SymbolIndex *Index = nullptr);

/// Get info about symbols at \p Pos.
std::vector<SymbolDetails> getSymbolInfo(ParsedAST &AST, Position Pos);

/// Find the record type references at \p Pos.
const CXXRecordDecl *findRecordTypeAt(ParsedAST &AST, Position Pos);

/// Given a record type declaration, find its base (parent) types.
std::vector<const CXXRecordDecl *> typeParents(const CXXRecordDecl *CXXRD);

/// Get type hierarchy information at \p Pos.
llvm::Optional<TypeHierarchyItem>
getTypeHierarchy(ParsedAST &AST, Position Pos, int Resolve,
                 TypeHierarchyDirection Direction);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_XREFS_H
