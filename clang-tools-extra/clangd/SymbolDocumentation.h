//===--- SymbolDocumentation.h ==---------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Class to parse doxygen comments into a flat structure for consumption
// in e.g. Hover and Code Completion
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SYMBOLDOCUMENTATION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SYMBOLDOCUMENTATION_H

#include "support/Markup.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <string>
#include <tuple>

namespace clang {
namespace clangd {

/// Contains pretty-printed type and desugared type
struct SymbolPrintedType {
  SymbolPrintedType() = default;
  SymbolPrintedType(const char *Type) : Type(Type) {}
  SymbolPrintedType(const char *Type, const char *AKAType)
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
struct SymbolParam {
  /// The printable parameter type, e.g. "int", or "typename" (in
  /// TemplateParameters), might be std::nullopt for macro parameters.
  std::optional<SymbolPrintedType> Type;
  /// std::nullopt for unnamed parameters.
  std::optional<std::string> Name;
  /// std::nullopt if no default is provided.
  std::optional<std::string> Default;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const SymbolPrintedType &T);
inline bool operator==(const SymbolPrintedType &LHS,
                       const SymbolPrintedType &RHS) {
  return std::tie(LHS.Type, LHS.AKA) == std::tie(RHS.Type, RHS.AKA);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &, const SymbolParam &);
inline bool operator==(const SymbolParam &LHS, const SymbolParam &RHS) {
  return std::tie(LHS.Type, LHS.Name, LHS.Default) ==
         std::tie(RHS.Type, RHS.Name, RHS.Default);
}

SymbolPrintedType printType(const TemplateTemplateParmDecl *TTP,
                            const PrintingPolicy &PP);
SymbolPrintedType printType(const NonTypeTemplateParmDecl *NTTP,
                            const PrintingPolicy &PP);
SymbolPrintedType printType(QualType QT, ASTContext &ASTCtx,
                            const PrintingPolicy &PP);

SymbolParam createSymbolParam(const ParmVarDecl *PVD, const PrintingPolicy &PP);

void fullCommentToMarkupDocument(
    markup::Document &Doc, const comments::FullComment *FC,
    const comments::CommandTraits &Traits,
    std::optional<SymbolPrintedType> SymbolType,
    std::optional<SymbolPrintedType> SymbolReturnType,
    const std::optional<std::vector<SymbolParam>> &SymbolParameters);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_SYMBOLDOCUMENTATION_H
