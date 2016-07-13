//===-- IncludeFixerContext.h - Include fixer context -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_INCLUDEFIXERCONTEXT_H
#define LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_INCLUDEFIXERCONTEXT_H

#include "find-all-symbols/SymbolInfo.h"
#include "clang/Tooling/Core/Replacement.h"
#include <string>
#include <vector>

namespace clang {
namespace include_fixer {

/// \brief A context for the symbol being queried.
class IncludeFixerContext {
public:
  IncludeFixerContext() = default;
  IncludeFixerContext(llvm::StringRef Name, llvm::StringRef ScopeQualifiers,
                      const std::vector<find_all_symbols::SymbolInfo> Symbols,
                      tooling::Range Range);

  /// \brief Create a replacement for adding missing namespace qualifiers to the
  /// symbol.
  tooling::Replacement createSymbolReplacement(llvm::StringRef FilePath,
                                               size_t Idx = 0);

  /// \brief Get symbol name.
  llvm::StringRef getSymbolIdentifier() const { return SymbolIdentifier; }

  /// \brief Get replacement range of the symbol.
  tooling::Range getSymbolRange() const { return SymbolRange; }

  /// \brief Get all matched symbols.
  const std::vector<find_all_symbols::SymbolInfo> &getMatchedSymbols() const {
    return MatchedSymbols;
  }

  /// \brief Get all headers. The headers are sorted in a descending order based
  /// on the popularity info in SymbolInfo.
  const std::vector<std::string> &getHeaders() const { return Headers; }

private:
  friend struct llvm::yaml::MappingTraits<IncludeFixerContext>;

  /// \brief The symbol name.
  std::string SymbolIdentifier;

  /// \brief The qualifiers of the scope in which SymbolIdentifier lookup
  /// occurs. It is represented as a sequence of names and scope resolution
  /// operatiors ::, ending with a scope resolution operator (e.g. a::b::).
  /// Empty if SymbolIdentifier is not in a specific scope.
  std::string SymbolScopedQualifiers;

  /// \brief The headers which have SymbolIdentifier definitions.
  std::vector<std::string> Headers;

  /// \brief The symbol candidates which match SymbolIdentifier. The symbols are
  /// sorted in a descending order based on the popularity info in SymbolInfo.
  std::vector<find_all_symbols::SymbolInfo> MatchedSymbols;

  /// \brief The replacement range of SymbolIdentifier.
  tooling::Range SymbolRange;
};

} // namespace include_fixer
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_INCLUDEFIXERCONTEXT_H
