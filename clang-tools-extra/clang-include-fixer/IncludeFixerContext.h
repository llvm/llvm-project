//===-- IncludeFixerContext.h - Include fixer context -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

/// A context for a file being processed. It includes all query
/// information, e.g. symbols being queried in database, all header candidates.
class IncludeFixerContext {
public:
  struct HeaderInfo {
    /// The header where QualifiedName comes from.
    std::string Header;
    /// A symbol name with completed namespace qualifiers which will
    /// replace the original symbol.
    std::string QualifiedName;
  };

  struct QuerySymbolInfo {
    /// The raw symbol name being queried in database. This name might
    /// miss some namespace qualifiers, and will be replaced by a fully
    /// qualified one.
    std::string RawIdentifier;

    /// The qualifiers of the scope in which SymbolIdentifier lookup
    /// occurs. It is represented as a sequence of names and scope resolution
    /// operators ::, ending with a scope resolution operator (e.g. a::b::).
    /// Empty if SymbolIdentifier is not in a specific scope.
    std::string ScopedQualifiers;

    /// The replacement range of RawIdentifier.
    tooling::Range Range;
  };

  IncludeFixerContext() = default;
  IncludeFixerContext(StringRef FilePath,
                      std::vector<QuerySymbolInfo> QuerySymbols,
                      std::vector<find_all_symbols::SymbolInfo> Symbols);

  /// Get symbol name.
  llvm::StringRef getSymbolIdentifier() const {
    return QuerySymbolInfos.front().RawIdentifier;
  }

  /// Get replacement range of the symbol.
  tooling::Range getSymbolRange() const {
    return QuerySymbolInfos.front().Range;
  }

  /// Get the file path to the file being processed.
  StringRef getFilePath() const { return FilePath; }

  /// Get header information.
  const std::vector<HeaderInfo> &getHeaderInfos() const { return HeaderInfos; }

  /// Get information of symbols being querid.
  const std::vector<QuerySymbolInfo> &getQuerySymbolInfos() const {
    return QuerySymbolInfos;
  }

private:
  friend struct llvm::yaml::MappingTraits<IncludeFixerContext>;

  /// The file path to the file being processed.
  std::string FilePath;

  /// All instances of an unidentified symbol being queried.
  std::vector<QuerySymbolInfo> QuerySymbolInfos;

  /// The symbol candidates which match SymbolIdentifier. The symbols are
  /// sorted in a descending order based on the popularity info in SymbolInfo.
  std::vector<find_all_symbols::SymbolInfo> MatchedSymbols;

  /// The header information.
  std::vector<HeaderInfo> HeaderInfos;
};

} // namespace include_fixer
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_INCLUDEFIXERCONTEXT_H
