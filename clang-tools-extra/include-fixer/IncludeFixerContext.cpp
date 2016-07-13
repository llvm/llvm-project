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

IncludeFixerContext::IncludeFixerContext(
    llvm::StringRef Name, llvm::StringRef ScopeQualifiers,
    const std::vector<find_all_symbols::SymbolInfo> Symbols,
    tooling::Range Range)
    : SymbolIdentifier(Name), SymbolScopedQualifiers(ScopeQualifiers),
      MatchedSymbols(Symbols), SymbolRange(Range) {
  // Deduplicate headers, so that we don't want to suggest the same header
  // twice.
  for (const auto &Symbol : MatchedSymbols)
    Headers.push_back(Symbol.getFilePath());
  Headers.erase(std::unique(Headers.begin(), Headers.end(),
                            [](const std::string &A, const std::string &B) {
                              return A == B;
                            }),
                Headers.end());
}

tooling::Replacement
IncludeFixerContext::createSymbolReplacement(llvm::StringRef FilePath,
                                             size_t Idx) {
  assert(Idx < MatchedSymbols.size());
  // No need to add missing qualifiers if SymbolIndentifer has a global scope
  // operator "::".
  if (getSymbolIdentifier().startswith("::"))
    return tooling::Replacement();
  std::string QualifiedName = MatchedSymbols[Idx].getQualifiedName();
  // For nested classes, the qualified name constructed from database misses
  // some stripped qualifiers, because when we search a symbol in database,
  // we strip qualifiers from the end until we find a result. So append the
  // missing stripped qualifiers here.
  //
  // Get stripped qualifiers.
  llvm::SmallVector<llvm::StringRef, 8> SymbolQualifiers;
  getSymbolIdentifier().split(SymbolQualifiers, "::");
  std::string StrippedQualifiers;
  while (!SymbolQualifiers.empty() &&
         !llvm::StringRef(QualifiedName).endswith(SymbolQualifiers.back())) {
    StrippedQualifiers = "::" + SymbolQualifiers.back().str();
    SymbolQualifiers.pop_back();
  }
  // Append the missing stripped qualifiers.
  std::string FullyQualifiedName = QualifiedName + StrippedQualifiers;
  auto pos = FullyQualifiedName.find(SymbolScopedQualifiers);
  return {FilePath, SymbolRange.getOffset(), SymbolRange.getLength(),
          FullyQualifiedName.substr(
              pos == std::string::npos ? 0 : SymbolScopedQualifiers.size())};
}

} // include_fixer
} // clang
