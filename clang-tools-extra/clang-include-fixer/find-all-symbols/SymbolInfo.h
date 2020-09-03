//===-- SymbolInfo.h - Symbol Info ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
/// Describes a named symbol from a header.
/// Symbols with the same qualified name and type (e.g. function overloads)
/// that appear in the same header are represented by a single SymbolInfo.
///
/// TODO: keep track of instances, e.g. overload locations and signatures.
class SymbolInfo {
public:
  /// The SymbolInfo Type.
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

  /// The Context Type.
  enum class ContextType {
    Namespace, // Symbols declared in a namespace.
    Record,    // Symbols declared in a class.
    EnumDecl,  // Enum constants declared in a enum declaration.
  };

  /// A pair of <ContextType, ContextName>.
  typedef std::pair<ContextType, std::string> Context;

  // Signals are signals gathered by observing how a symbol is used.
  // These are used to rank results.
  struct Signals {
    Signals() {}
    Signals(unsigned Seen, unsigned Used) : Seen(Seen), Used(Used) {}

    // Number of times this symbol was visible to a TU.
    unsigned Seen = 0;

    // Number of times this symbol was referenced a TU's main file.
    unsigned Used = 0;

    Signals &operator+=(const Signals &RHS);
    Signals operator+(const Signals &RHS) const;
    bool operator==(const Signals &RHS) const;
  };

  using SignalMap = std::map<SymbolInfo, Signals>;

  // The default constructor is required by YAML traits in
  // LLVM_YAML_IS_DOCUMENT_LIST_VECTOR.
  SymbolInfo() : Type(SymbolKind::Unknown) {}

  SymbolInfo(llvm::StringRef Name, SymbolKind Type, llvm::StringRef FilePath,
             const std::vector<Context> &Contexts);

  void SetFilePath(llvm::StringRef Path) { FilePath = std::string(Path); }

  /// Get symbol name.
  llvm::StringRef getName() const { return Name; }

  /// Get the fully-qualified symbol name.
  std::string getQualifiedName() const;

  /// Get symbol type.
  SymbolKind getSymbolKind() const { return Type; }

  /// Get a relative file path where symbol comes from.
  llvm::StringRef getFilePath() const { return FilePath; }

  /// Get symbol contexts.
  const std::vector<SymbolInfo::Context> &getContexts() const {
    return Contexts;
  }

  bool operator<(const SymbolInfo &Symbol) const;

  bool operator==(const SymbolInfo &Symbol) const;

private:
  friend struct llvm::yaml::MappingTraits<struct SymbolAndSignals>;

  /// Identifier name.
  std::string Name;

  /// Symbol type.
  SymbolKind Type;

  /// The file path where the symbol comes from. It's a relative file
  /// path based on the build directory.
  std::string FilePath;

  /// Contains information about symbol contexts. Context information is
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
};

struct SymbolAndSignals {
  SymbolInfo Symbol;
  SymbolInfo::Signals Signals;
  bool operator==(const SymbolAndSignals& RHS) const;
};

/// Write SymbolInfos to a stream (YAML format).
bool WriteSymbolInfosToStream(llvm::raw_ostream &OS,
                              const SymbolInfo::SignalMap &Symbols);

/// Read SymbolInfos from a YAML document.
std::vector<SymbolAndSignals> ReadSymbolInfosFromYAML(llvm::StringRef Yaml);

} // namespace find_all_symbols
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_FIND_ALL_SYMBOLS_SYMBOLINFO_H
