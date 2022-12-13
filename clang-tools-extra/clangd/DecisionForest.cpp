//===--- DecisionForest.cpp --------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Features.inc"

#if !CLANGD_DECISION_FOREST
#include "Quality.h"
#include <cstdlib>

namespace clang {
namespace clangd {
DecisionForestScores
evaluateDecisionForest(const SymbolQualitySignals &Quality,
                       const SymbolRelevanceSignals &Relevance, float Base) {
  llvm::errs() << "Clangd was compiled without decision forest support.\n";
  std::abort();
}

} // namespace clangd
} // namespace clang

#else // !CLANGD_DECISION_FOREST

#include "CompletionModel.h"
#include "Quality.h"
#include <cmath>

namespace clang {
namespace clangd {

DecisionForestScores
evaluateDecisionForest(const SymbolQualitySignals &Quality,
                       const SymbolRelevanceSignals &Relevance, float Base) {
  Example E;
  E.setIsDeprecated(Quality.Deprecated);
  E.setIsReservedName(Quality.ReservedName);
  E.setIsImplementationDetail(Quality.ImplementationDetail);
  E.setNumReferences(Quality.References);
  E.setSymbolCategory(Quality.Category);

  SymbolRelevanceSignals::DerivedSignals Derived =
      Relevance.calculateDerivedSignals();
  int NumMatch = 0;
  if (Relevance.ContextWords) {
    for (const auto &Word : Relevance.ContextWords->keys()) {
      if (Relevance.Name.contains_insensitive(Word)) {
        ++NumMatch;
      }
    }
  }
  E.setIsNameInContext(NumMatch > 0);
  E.setNumNameInContext(NumMatch);
  E.setFractionNameInContext(
      Relevance.ContextWords && !Relevance.ContextWords->empty()
          ? NumMatch * 1.0 / Relevance.ContextWords->size()
          : 0);
  E.setIsInBaseClass(Relevance.InBaseClass);
  E.setFileProximityDistanceCost(Derived.FileProximityDistance);
  E.setSemaFileProximityScore(Relevance.SemaFileProximityScore);
  E.setSymbolScopeDistanceCost(Derived.ScopeProximityDistance);
  E.setSemaSaysInScope(Relevance.SemaSaysInScope);
  E.setScope(Relevance.Scope);
  E.setContextKind(Relevance.Context);
  E.setIsInstanceMember(Relevance.IsInstanceMember);
  E.setHadContextType(Relevance.HadContextType);
  E.setHadSymbolType(Relevance.HadSymbolType);
  E.setTypeMatchesPreferred(Relevance.TypeMatchesPreferred);

  DecisionForestScores Scores;
  // Exponentiating DecisionForest prediction makes the score of each tree a
  // multiplciative boost (like NameMatch). This allows us to weigh the
  // prediction score and NameMatch appropriately.
  Scores.ExcludingName = pow(Base, Evaluate(E));
  // Following cases are not part of the generated training dataset:
  //  - Symbols with `NeedsFixIts`.
  //  - Forbidden symbols.
  //  - Keywords: Dataset contains only macros and decls.
  if (Relevance.NeedsFixIts)
    Scores.ExcludingName *= 0.5;
  if (Relevance.Forbidden)
    Scores.ExcludingName *= 0;
  if (Quality.Category == SymbolQualitySignals::Keyword)
    Scores.ExcludingName *= 4;

  // NameMatch should be a multiplier on total score to support rescoring.
  Scores.Total = Relevance.NameMatch * Scores.ExcludingName;
  return Scores;
}

} // namespace clangd
} // namespace clang

#endif // !CLANGD_DECISION_FOREST
