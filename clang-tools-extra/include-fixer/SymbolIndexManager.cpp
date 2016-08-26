//===-- SymbolIndexManager.cpp - Managing multiple SymbolIndices-*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolIndexManager.h"
#include "find-all-symbols/SymbolInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "include-fixer"

namespace clang {
namespace include_fixer {

using clang::find_all_symbols::SymbolInfo;

/// Sorts SymbolInfos based on the popularity info in SymbolInfo.
static void rankByPopularity(std::vector<SymbolInfo> &Symbols) {
  // First collect occurrences per header file.
  llvm::DenseMap<llvm::StringRef, unsigned> HeaderPopularity;
  for (const SymbolInfo &Symbol : Symbols) {
    unsigned &Popularity = HeaderPopularity[Symbol.getFilePath()];
    Popularity = std::max(Popularity, Symbol.getNumOccurrences());
  }

  // Sort by the gathered popularities. Use file name as a tie breaker so we can
  // deduplicate.
  std::sort(Symbols.begin(), Symbols.end(),
            [&](const SymbolInfo &A, const SymbolInfo &B) {
              auto APop = HeaderPopularity[A.getFilePath()];
              auto BPop = HeaderPopularity[B.getFilePath()];
              if (APop != BPop)
                return APop > BPop;
              return A.getFilePath() < B.getFilePath();
            });
}

std::vector<find_all_symbols::SymbolInfo>
SymbolIndexManager::search(llvm::StringRef Identifier,
                           bool IsNestedSearch) const {
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
  std::vector<clang::find_all_symbols::SymbolInfo> MatchedSymbols;
  do {
    std::vector<clang::find_all_symbols::SymbolInfo> Symbols;
    for (const auto &DB : SymbolIndices) {
      auto Res = DB->search(Names.back().str());
      Symbols.insert(Symbols.end(), Res.begin(), Res.end());
    }

    DEBUG(llvm::dbgs() << "Searching " << Names.back() << "... got "
                       << Symbols.size() << " results...\n");

    for (const auto &Symbol : Symbols) {
      // Match the identifier name without qualifier.
      if (Symbol.getName() == Names.back()) {
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

          MatchedSymbols.push_back(Symbol);
        }
      }
    }
    Names.pop_back();
    TookPrefix = true;
  } while (MatchedSymbols.empty() && !Names.empty() && IsNestedSearch);

  rankByPopularity(MatchedSymbols);
  return MatchedSymbols;
}

} // namespace include_fixer
} // namespace clang
