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
  struct HeaderInfo {
    /// \brief The header where QualifiedName comes from.
    std::string Header;
    /// \brief A symbol name with completed namespace qualifiers which will
    /// replace the original symbol.
    std::string QualifiedName;
  };

  struct QuerySymbolInfo {
    /// \brief The raw symbol name being queried in database. This name might
    /// miss some namespace qualifiers, and will be replaced by a fully
    /// qualified one.
    std::string RawIdentifier;

    /// \brief The qualifiers of the scope in which SymbolIdentifier lookup
    /// occurs. It is represented as a sequence of names and scope resolution
    /// operatiors ::, ending with a scope resolution operator (e.g. a::b::).
    /// Empty if SymbolIdentifier is not in a specific scope.
    std::string ScopedQualifiers;

    /// \brief The replacement range of RawIdentifier.
    tooling::Range Range;
  };

  IncludeFixerContext() = default;
  IncludeFixerContext(const QuerySymbolInfo &QuerySymbol,
                      const std::vector<find_all_symbols::SymbolInfo> Symbols);

  /// \brief Get symbol name.
  llvm::StringRef getSymbolIdentifier() const {
    return QuerySymbol.RawIdentifier;
  }

  /// \brief Get replacement range of the symbol.
  tooling::Range getSymbolRange() const { return QuerySymbol.Range; }

  const std::vector<HeaderInfo> &getHeaderInfos() const { return HeaderInfos; }

private:
  friend struct llvm::yaml::MappingTraits<IncludeFixerContext>;

  /// \brief The symbol candidates which match SymbolIdentifier. The symbols are
  /// sorted in a descending order based on the popularity info in SymbolInfo.
  std::vector<find_all_symbols::SymbolInfo> MatchedSymbols;

  /// \brief The header information.
  std::vector<HeaderInfo> HeaderInfos;

  /// \brief The information of the symbol being queried.
  QuerySymbolInfo QuerySymbol;
};

} // namespace include_fixer
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_INCLUDEFIXERCONTEXT_H
