//===-- SymbolInfo.h - Symbol Info ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_FIND_ALL_SYMBOLS_SYMBOLINFO_H
#define LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_FIND_ALL_SYMBOLS_SYMBOLINFO_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
#include <string>
#include <vector>

namespace clang {
namespace find_all_symbols {

/// \brief Contains all information for a Symbol.
class SymbolInfo {
public:
  /// \brief The SymbolInfo Type.
  enum class SymbolKind {
    Function,
    Class,
    Variable,
    TypedefName,
    EnumDecl,
    EnumConstantDecl,
    Macro,
    Unknown,
  };

  /// \brief The Context Type.
  enum class ContextType {
    Namespace, // Symbols declared in a namespace.
    Record,    // Symbols declared in a class.
    EnumDecl,  // Enum constants declared in a enum declaration.
  };

  /// \brief A pair of <ContextType, ContextName>.
  typedef std::pair<ContextType, std::string> Context;

  // The default constructor is required by YAML traits in
  // LLVM_YAML_IS_DOCUMENT_LIST_VECTOR.
  SymbolInfo() : Type(SymbolKind::Unknown), LineNumber(-1) {}

  SymbolInfo(llvm::StringRef Name, SymbolKind Type, llvm::StringRef FilePath,
             int LineNumber, const std::vector<Context> &Contexts,
             unsigned NumOccurrences = 0);

  void SetFilePath(llvm::StringRef Path) { FilePath = Path; }

  /// \brief Get symbol name.
  llvm::StringRef getName() const { return Name; }

  /// \brief Get the fully-qualified symbol name.
  std::string getQualifiedName() const;

  /// \brief Get symbol type.
  SymbolKind getSymbolKind() const { return Type; }

  /// \brief Get a relative file path where symbol comes from.
  llvm::StringRef getFilePath() const { return FilePath; }

  /// \brief Get symbol contexts.
  const std::vector<SymbolInfo::Context> &getContexts() const {
    return Contexts;
  }

  /// \brief Get a 1-based line number of the symbol's declaration.
  int getLineNumber() const { return LineNumber; }

  /// \brief The number of times this symbol was found during an indexing run.
  unsigned getNumOccurrences() const { return NumOccurrences; }

  bool operator<(const SymbolInfo &Symbol) const;

  bool operator==(const SymbolInfo &Symbol) const;

private:
  friend struct llvm::yaml::MappingTraits<SymbolInfo>;

  /// \brief Identifier name.
  std::string Name;

  /// \brief Symbol type.
  SymbolKind Type;

  /// \brief The file path where the symbol comes from. It's a relative file
  /// path based on the build directory.
  std::string FilePath;

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

  /// \brief The number of times this symbol was found during an indexing
  /// run. Populated by the reducer and used to rank results.
  unsigned NumOccurrences;
};

/// \brief Write SymbolInfos to a stream (YAML format).
bool WriteSymbolInfosToStream(llvm::raw_ostream &OS,
                              const std::set<SymbolInfo> &Symbols);

/// \brief Read SymbolInfos from a YAML document.
std::vector<SymbolInfo> ReadSymbolInfosFromYAML(llvm::StringRef Yaml);

} // namespace find_all_symbols
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_FIND_ALL_SYMBOLS_SYMBOLINFO_H
