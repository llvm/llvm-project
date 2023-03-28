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
#include <optional>
#include <string>
#include <vector>

namespace clang {
namespace clangd {

/// Contains detailed information about a Symbol. Especially useful when
/// generating hover responses. It can be rendered as a hover panel, or
/// embedding clients can use the structured information to provide their own
/// UI.
struct HoverInfo {
  /// Contains pretty-printed type and desugared type
  struct PrintedType {
    PrintedType() = default;
    PrintedType(const char *Type) : Type(Type) {}
    PrintedType(const char *Type, const char *AKAType)
        : Type(Type), AKA(AKAType) {}

    /// Pretty-printed type
    std::string Type;
    /// Desugared type
    std::optional<std::string> AKA;
  };

  /// Represents parameters of a function, a template or a macro.
  /// For example:
  /// - void foo(ParamType Name = DefaultValue)
  /// - #define FOO(Name)
  /// - template <ParamType Name = DefaultType> class Foo {};
  struct Param {
    /// The printable parameter type, e.g. "int", or "typename" (in
    /// TemplateParameters), might be std::nullopt for macro parameters.
    std::optional<PrintedType> Type;
    /// std::nullopt for unnamed parameters.
    std::optional<std::string> Name;
    /// std::nullopt if no default is provided.
    std::optional<std::string> Default;
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
  std::optional<std::string> NamespaceScope;
  /// Remaining named contexts in symbol's qualified name, empty string means
  /// symbol is not local.
  std::string LocalScope;
  /// Name of the symbol, does not contain any "::".
  std::string Name;
  /// Header providing the symbol (best match). Contains ""<>.
  std::string Provider;
  std::optional<Range> SymRange;
  index::SymbolKind Kind = index::SymbolKind::Unknown;
  std::string Documentation;
  /// Source code containing the definition of the symbol.
  std::string Definition;
  const char *DefinitionLanguage = "cpp";
  /// Access specifier for declarations inside class/struct/unions, empty for
  /// others.
  std::string AccessSpecifier;
  /// Printable variable type.
  /// Set only for variables.
  std::optional<PrintedType> Type;
  /// Set for functions and lambdas.
  std::optional<PrintedType> ReturnType;
  /// Set for functions, lambdas and macros with parameters.
  std::optional<std::vector<Param>> Parameters;
  /// Set for all templates(function, class, variable).
  std::optional<std::vector<Param>> TemplateParameters;
  /// Contains the evaluated value of the symbol if available.
  std::optional<std::string> Value;
  /// Contains the byte-size of fields and types where it's interesting.
  std::optional<uint64_t> Size;
  /// Contains the offset of fields within the enclosing class.
  std::optional<uint64_t> Offset;
  /// Contains the padding following a field within the enclosing class.
  std::optional<uint64_t> Padding;
  // Set when symbol is inside function call. Contains information extracted
  // from the callee definition about the argument this is passed as.
  std::optional<Param> CalleeArgInfo;
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
  std::optional<PassType> CallPassType;
  // Filled when hovering over the #include line. Contains the names of symbols
  // from a #include'd file that are used in the main file, sorted in
  // alphabetical order.
  std::vector<std::string> UsedSymbolNames;

  /// Produce a user-readable information.
  markup::Document present() const;
};

inline bool operator==(const HoverInfo::PrintedType &LHS,
                       const HoverInfo::PrintedType &RHS) {
  return std::tie(LHS.Type, LHS.AKA) == std::tie(RHS.Type, RHS.AKA);
}

inline bool operator==(const HoverInfo::PassType &LHS,
                       const HoverInfo::PassType &RHS) {
  return std::tie(LHS.PassBy, LHS.Converted) ==
         std::tie(RHS.PassBy, RHS.Converted);
}

// Try to infer structure of a documentation comment (e.g. line breaks).
// FIXME: move to another file so CodeComplete doesn't depend on Hover.
void parseDocumentation(llvm::StringRef Input, markup::Document &Output);

llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                              const HoverInfo::PrintedType &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const HoverInfo::Param &);
inline bool operator==(const HoverInfo::Param &LHS,
                       const HoverInfo::Param &RHS) {
  return std::tie(LHS.Type, LHS.Name, LHS.Default) ==
         std::tie(RHS.Type, RHS.Name, RHS.Default);
}

/// Get the hover information when hovering at \p Pos.
std::optional<HoverInfo> getHover(ParsedAST &AST, Position Pos,
                                  const format::FormatStyle &Style,
                                  const SymbolIndex *Index);

} // namespace clangd
} // namespace clang

#endif
