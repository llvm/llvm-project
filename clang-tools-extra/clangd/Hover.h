//===--- Hover.h - Information about code at the cursor location -*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_HOVER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_HOVER_H

#include "ParsedAST.h"
#include "Protocol.h"
#include "support/Markup.h"
#include "clang/Index/IndexSymbol.h"

namespace clang {
namespace clangd {

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
    /// TemplateParameters), might be None for macro parameters.
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
  index::SymbolKind Kind = index::SymbolKind::Unknown;
  std::string Documentation;
  /// Source code containing the definition of the symbol.
  std::string Definition;

  /// Access specifier for declarations inside class/struct/unions, empty for
  /// others.
  std::string AccessSpecifier;
  /// Pretty-printed variable type.
  /// Set only for variables.
  llvm::Optional<std::string> Type;
  /// Set for functions and lambdas.
  llvm::Optional<std::string> ReturnType;
  /// Set for functions, lambdas and macros with parameters.
  llvm::Optional<std::vector<Param>> Parameters;
  /// Set for all templates(function, class, variable).
  llvm::Optional<std::vector<Param>> TemplateParameters;
  /// Contains the evaluated value of the symbol if available.
  llvm::Optional<std::string> Value;
  /// Contains the byte-size of fields and types where it's interesting.
  llvm::Optional<uint64_t> Size;
  /// Contains the offset of fields within the enclosing class.
  llvm::Optional<uint64_t> Offset;
  // Set when symbol is inside function call. Contains information extracted
  // from the callee definition about the argument this is passed as.
  llvm::Optional<Param> CalleeArgInfo;
  struct PassType {
    // How the variable is passed to callee.
    enum PassMode { Ref, ConstRef, Value };
    PassMode PassBy = Ref;
    // True if type conversion happened. This includes calls to implicit
    // constructor, as well as built-in type conversions. Casting to base class
    // is not considered conversion.
    bool Converted = false;
  };
  // Set only if CalleeArgInfo is set.
  llvm::Optional<PassType> CallPassType;

  /// Produce a user-readable information.
  markup::Document present() const;
};

inline bool operator==(const HoverInfo::PassType &LHS,
                       const HoverInfo::PassType &RHS) {
  return std::tie(LHS.PassBy, LHS.Converted) ==
         std::tie(RHS.PassBy, RHS.Converted);
}

// Try to infer structure of a documentation comment (e.g. line breaks).
// FIXME: move to another file so CodeComplete doesn't depend on Hover.
void parseDocumentation(llvm::StringRef Input, markup::Document &Output);

llvm::raw_ostream &operator<<(llvm::raw_ostream &, const HoverInfo::Param &);
inline bool operator==(const HoverInfo::Param &LHS,
                       const HoverInfo::Param &RHS) {
  return std::tie(LHS.Type, LHS.Name, LHS.Default) ==
         std::tie(RHS.Type, RHS.Name, RHS.Default);
}

/// Get the hover information when hovering at \p Pos.
llvm::Optional<HoverInfo> getHover(ParsedAST &AST, Position Pos,
                                   format::FormatStyle Style,
                                   const SymbolIndex *Index);

} // namespace clangd
} // namespace clang

#endif
