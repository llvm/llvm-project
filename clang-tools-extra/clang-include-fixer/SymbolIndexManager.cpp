//===-- SymbolIndexManager.cpp - Managing multiple SymbolIndices-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolIndexManager.h"
#include "find-all-symbols/SymbolInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "clang-include-fixer"

namespace clang {
namespace include_fixer {

using find_all_symbols::SymbolInfo;
using find_all_symbols::SymbolAndSignals;

// Calculate a score based on whether we think the given header is closely
// related to the given source file.
static double similarityScore(llvm::StringRef FileName,
                              llvm::StringRef Header) {
  // Compute the maximum number of common path segments between Header and
  // a suffix of FileName.
  // We do not do a full longest common substring computation, as Header
  // specifies the path we would directly #include, so we assume it is rooted
  // relatively to a subproject of the repository.
  int MaxSegments = 1;
  for (auto FileI = llvm::sys::path::begin(FileName),
            FileE = llvm::sys::path::end(FileName);
       FileI != FileE; ++FileI) {
    int Segments = 0;
    for (auto HeaderI = llvm::sys::path::begin(Header),
              HeaderE = llvm::sys::path::end(Header), I = FileI;
         HeaderI != HeaderE && *I == *HeaderI && I != FileE; ++I, ++HeaderI) {
      ++Segments;
    }
    MaxSegments = std::max(Segments, MaxSegments);
  }
  return MaxSegments;
}

static void rank(std::vector<SymbolAndSignals> &Symbols,
                 llvm::StringRef FileName) {
  llvm::StringMap<double> Score;
  for (const auto &Symbol : Symbols) {
    // Calculate a score from the similarity of the header the symbol is in
    // with the current file and the popularity of the symbol.
    double NewScore = similarityScore(FileName, Symbol.Symbol.getFilePath()) *
                      (1.0 + std::log2(1 + Symbol.Signals.Seen));
    double &S = Score[Symbol.Symbol.getFilePath()];
    S = std::max(S, NewScore);
  }
  // Sort by the gathered scores. Use file name as a tie breaker so we can
  // deduplicate.
  llvm::sort(Symbols,
             [&](const SymbolAndSignals &A, const SymbolAndSignals &B) {
               auto AS = Score[A.Symbol.getFilePath()];
               auto BS = Score[B.Symbol.getFilePath()];
               if (AS != BS)
                 return AS > BS;
               return A.Symbol.getFilePath() < B.Symbol.getFilePath();
             });
}

std::vector<find_all_symbols::SymbolInfo>
SymbolIndexManager::search(llvm::StringRef Identifier,
                           bool IsNestedSearch,
                           llvm::StringRef FileName) const {
  // The identifier may be fully qualified, so split it and get all the context
  // names.
  llvm::SmallVector<llvm::StringRef, 8> Names;
  Identifier.split(Names, "::");

  bool IsFullyQualified = false;
  if (Identifier.startswith("::")) {
    Names.erase(Names.begin()); // Drop first (empty) element.
    IsFullyQualified = true;
  }

  // As long as we don't find a result keep stripping name parts from the end.
  // This is to support nested classes which aren't recorded in the database.
  // Eventually we will either hit a class (namespaces aren't in the database
  // either) and can report that result.
  bool TookPrefix = false;
  std::vector<SymbolAndSignals> MatchedSymbols;
  do {
    std::vector<SymbolAndSignals> Symbols;
    for (const auto &DB : SymbolIndices) {
      auto Res = DB.get()->search(Names.back());
      Symbols.insert(Symbols.end(), Res.begin(), Res.end());
    }

    LLVM_DEBUG(llvm::dbgs() << "Searching " << Names.back() << "... got "
                            << Symbols.size() << " results...\n");

    for (auto &SymAndSig : Symbols) {
      const SymbolInfo &Symbol = SymAndSig.Symbol;
      // Match the identifier name without qualifier.
      bool IsMatched = true;
      auto SymbolContext = Symbol.getContexts().begin();
      auto IdentiferContext = Names.rbegin() + 1; // Skip identifier name.
      // Match the remaining context names.
      while (IdentiferContext != Names.rend() &&
             SymbolContext != Symbol.getContexts().end()) {
        if (SymbolContext->second == *IdentiferContext) {
          ++IdentiferContext;
          ++SymbolContext;
        } else if (SymbolContext->first ==
                   find_all_symbols::SymbolInfo::ContextType::EnumDecl) {
          // Skip non-scoped enum context.
          ++SymbolContext;
        } else {
          IsMatched = false;
          break;
        }
      }

      // If the name was qualified we only want to add results if we evaluated
      // all contexts.
      if (IsFullyQualified)
        IsMatched &= (SymbolContext == Symbol.getContexts().end());

      // FIXME: Support full match. At this point, we only find symbols in
      // database which end with the same contexts with the identifier.
      if (IsMatched && IdentiferContext == Names.rend()) {
        // If we're in a situation where we took a prefix but the thing we
        // found couldn't possibly have a nested member ignore it.
        if (TookPrefix &&
            (Symbol.getSymbolKind() == SymbolInfo::SymbolKind::Function ||
             Symbol.getSymbolKind() == SymbolInfo::SymbolKind::Variable ||
             Symbol.getSymbolKind() ==
                 SymbolInfo::SymbolKind::EnumConstantDecl ||
             Symbol.getSymbolKind() == SymbolInfo::SymbolKind::Macro))
          continue;

        MatchedSymbols.push_back(std::move(SymAndSig));
      }
    }
    Names.pop_back();
    TookPrefix = true;
  } while (MatchedSymbols.empty() && !Names.empty() && IsNestedSearch);

  rank(MatchedSymbols, FileName);
  // Strip signals, they are no longer needed.
  std::vector<SymbolInfo> Res;
  Res.reserve(MatchedSymbols.size());
  for (auto &SymAndSig : MatchedSymbols)
    Res.push_back(std::move(SymAndSig.Symbol));
  return Res;
}

} // namespace include_fixer
} // namespace clang
