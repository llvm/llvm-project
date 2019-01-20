//===-- IncludeFixerContext.cpp - Include fixer context ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncludeFixerContext.h"
#include <algorithm>

namespace clang {
namespace include_fixer {

namespace {

// Splits a multiply qualified names (e.g. a::b::c).
llvm::SmallVector<llvm::StringRef, 8>
SplitQualifiers(llvm::StringRef StringQualifiers) {
  llvm::SmallVector<llvm::StringRef, 8> Qualifiers;
  StringQualifiers.split(Qualifiers, "::");
  return Qualifiers;
}

std::string createQualifiedNameForReplacement(
    llvm::StringRef RawSymbolName,
    llvm::StringRef SymbolScopedQualifiersName,
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
  auto SymbolQualifiers = SplitQualifiers(RawSymbolName);
  std::string StrippedQualifiers;
  while (!SymbolQualifiers.empty() &&
         !llvm::StringRef(QualifiedName).endswith(SymbolQualifiers.back())) {
    StrippedQualifiers =
        "::" + SymbolQualifiers.back().str() + StrippedQualifiers;
    SymbolQualifiers.pop_back();
  }
  // Append the missing stripped qualifiers.
  std::string FullyQualifiedName = QualifiedName + StrippedQualifiers;

  // Try to find and skip the common prefix qualifiers.
  auto FullySymbolQualifiers = SplitQualifiers(FullyQualifiedName);
  auto ScopedQualifiers = SplitQualifiers(SymbolScopedQualifiersName);
  auto FullySymbolQualifiersIter = FullySymbolQualifiers.begin();
  auto SymbolScopedQualifiersIter = ScopedQualifiers.begin();
  while (FullySymbolQualifiersIter != FullySymbolQualifiers.end() &&
         SymbolScopedQualifiersIter != ScopedQualifiers.end()) {
    if (*FullySymbolQualifiersIter != *SymbolScopedQualifiersIter)
      break;
    ++FullySymbolQualifiersIter;
    ++SymbolScopedQualifiersIter;
  }
  std::string Result;
  for (; FullySymbolQualifiersIter != FullySymbolQualifiers.end();
       ++FullySymbolQualifiersIter) {
    if (!Result.empty())
      Result += "::";
    Result += *FullySymbolQualifiersIter;
  }
  return Result;
}

} // anonymous namespace

IncludeFixerContext::IncludeFixerContext(
    StringRef FilePath, std::vector<QuerySymbolInfo> QuerySymbols,
    std::vector<find_all_symbols::SymbolInfo> Symbols)
    : FilePath(FilePath), QuerySymbolInfos(std::move(QuerySymbols)),
      MatchedSymbols(std::move(Symbols)) {
  // Remove replicated QuerySymbolInfos with the same range.
  //
  // QuerySymbolInfos may contain replicated elements. Because CorrectTypo
  // callback doesn't always work as we expected. In somecases, it will be
  // triggered at the same position or unidentified symbol multiple times.
  std::sort(QuerySymbolInfos.begin(), QuerySymbolInfos.end(),
            [&](const QuerySymbolInfo &A, const QuerySymbolInfo &B) {
              return std::make_pair(A.Range.getOffset(), A.Range.getLength()) <
                     std::make_pair(B.Range.getOffset(), B.Range.getLength());
            });
  QuerySymbolInfos.erase(
      std::unique(QuerySymbolInfos.begin(), QuerySymbolInfos.end(),
                  [](const QuerySymbolInfo &A, const QuerySymbolInfo &B) {
                    return A.Range == B.Range;
                  }),
      QuerySymbolInfos.end());
  for (const auto &Symbol : MatchedSymbols) {
    HeaderInfos.push_back(
        {Symbol.getFilePath().str(),
         createQualifiedNameForReplacement(
             QuerySymbolInfos.front().RawIdentifier,
             QuerySymbolInfos.front().ScopedQualifiers, Symbol)});
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
