//===-- SymbolInfo.h - find all symbols--------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_FIND_ALL_SYMBOLS_SYMBOL_INFO_H
#define LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_FIND_ALL_SYMBOLS_SYMBOL_INFO_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include <set>
#include <string>
#include <vector>

namespace clang {
namespace find_all_symbols {

/// \brief Contains all information for a Symbol.
struct SymbolInfo {
  enum SymbolKind {
    Function,
    Class,
    Variable,
    TypedefName,
  };

  enum ContextType {
    Namespace, // Symbols declared in a namespace.
    Record,    // Symbols declared in a class.
  };

  /// \brief Identifier name.
  std::string Name;

  /// \brief Symbol type.
  SymbolKind Type;

  /// \brief The file path where the symbol comes from.
  std::string FilePath;

  /// \brief A pair of <ContextType, ContextName>.
  typedef std::pair<ContextType, std::string> Context;

  /// \brief Contains information about symbol contexts. Context information is
  /// stored from the inner-most level to outer-most level.
  ///
  /// For example, if a symbol 'x' is declared as:
  ///     namespace na { namespace nb { class A { int x; } } }
  /// The contexts would be { {RECORD, "A"}, {NAMESPACE, "nb"}, {NAMESPACE,
  /// "na"} }.
  /// The name of an anonymous namespace is "".
  ///
  /// If the symbol is declared in `TranslationUnitDecl`, it has no context.
  std::vector<Context> Contexts;

  /// \brief The 1-based line number of of the symbol's declaration.
  int LineNumber;

  struct FunctionInfo {
    std::string ReturnType;
    std::vector<std::string> ParameterTypes;
  };

  struct TypedefNameInfo {
    std::string UnderlyingType;
  };

  struct VariableInfo {
    std::string Type;
  };

  /// \brief The function information.
  llvm::Optional<FunctionInfo> FunctionInfos;

  /// \brief The typedef information.
  llvm::Optional<TypedefNameInfo> TypedefNameInfos;

  /// \brief The variable information.
  llvm::Optional<VariableInfo> VariableInfos;

  bool operator==(const SymbolInfo &Symbol) const;

  bool operator<(const SymbolInfo &Symbol) const;
};

/// \brief Write SymbolInfos to a single file (YAML format).
bool WriteSymboInfosToFile(llvm::StringRef FilePath,
                           const std::set<SymbolInfo> &Symbols);

/// \brief Read SymbolInfos from a YAML document.
std::vector<SymbolInfo> ReadSymbolInfosFromYAML(llvm::StringRef Yaml);

} // namespace find_all_symbols
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_FIND_ALL_SYMBOLS_SYMBOL_INFO_H
