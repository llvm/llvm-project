//===-- IncludeFixerContext.cpp - Include fixer context ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IncludeFixerContext.h"
#include <algorithm>

namespace clang {
namespace include_fixer {

namespace {

std::string createQualifiedNameForReplacement(
    llvm::StringRef RawSymbolName,
    llvm::StringRef SymbolScopedQualifiers,
    const find_all_symbols::SymbolInfo &MatchedSymbol) {
  // No need to add missing qualifiers if SymbolIndentifer has a global scope
  // operator "::".
  if (RawSymbolName.startswith("::"))
    return RawSymbolName;

  std::string QualifiedName = MatchedSymbol.getQualifiedName();

  // For nested classes, the qualified name constructed from database misses
  // some stripped qualifiers, because when we search a symbol in database,
  // we strip qualifiers from the end until we find a result. So append the
  // missing stripped qualifiers here.
  //
  // Get stripped qualifiers.
  llvm::SmallVector<llvm::StringRef, 8> SymbolQualifiers;
  RawSymbolName.split(SymbolQualifiers, "::");
  std::string StrippedQualifiers;
  while (!SymbolQualifiers.empty() &&
         !llvm::StringRef(QualifiedName).endswith(SymbolQualifiers.back())) {
    StrippedQualifiers = "::" + SymbolQualifiers.back().str();
    SymbolQualifiers.pop_back();
  }
  // Append the missing stripped qualifiers.
  std::string FullyQualifiedName = QualifiedName + StrippedQualifiers;

  // Skips symbol scoped qualifiers prefix.
  if (llvm::StringRef(FullyQualifiedName).startswith(SymbolScopedQualifiers))
    return FullyQualifiedName.substr(SymbolScopedQualifiers.size());

  return FullyQualifiedName;
}

} // anonymous namespace

IncludeFixerContext::IncludeFixerContext(
    llvm::StringRef Name, llvm::StringRef ScopeQualifiers,
    std::vector<find_all_symbols::SymbolInfo> Symbols,
    tooling::Range Range)
    : SymbolIdentifier(Name), SymbolScopedQualifiers(ScopeQualifiers),
      MatchedSymbols(std::move(Symbols)), SymbolRange(Range) {
  for (const auto &Symbol : MatchedSymbols) {
    HeaderInfos.push_back({Symbol.getFilePath().str(),
                           createQualifiedNameForReplacement(
                               SymbolIdentifier, ScopeQualifiers, Symbol)});
  }
  // Deduplicate header infos.
  HeaderInfos.erase(std::unique(HeaderInfos.begin(), HeaderInfos.end(),
                                [](const HeaderInfo &A, const HeaderInfo &B) {
                                  return A.Header == B.Header &&
                                         A.QualifiedName == B.QualifiedName;
                                }),
                    HeaderInfos.end());
}

} // include_fixer
} // clang
