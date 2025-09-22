//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_SUSPICIOUSCALLARGUMENTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_SUSPICIOUSCALLARGUMENTCHECK_H

#include "../ClangTidyCheck.h"
#include "llvm/ADT/StringSet.h"
#include <optional>

namespace clang::tidy::readability {

/// Finds function calls where the arguments passed are provided out of order,
/// based on the difference between the argument name and the parameter names
/// of the function.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability/suspicious-call-argument.html
class SuspiciousCallArgumentCheck : public ClangTidyCheck {
  enum class Heuristic {
    Equality,
    Abbreviation,
    Prefix,
    Suffix,
    Substring,
    Levenshtein,
    JaroWinkler,
    Dice
  };

  /// When applying a heuristic, the value of this enum decides which kind of
  /// bound will be selected from the bounds configured for the heuristic.
  /// This only applies to heuristics that can take bounds.
  enum class BoundKind {
    /// Check for dissimilarity of the names. Names are deemed dissimilar if
    /// the similarity measurement is **below** the configured threshold.
    DissimilarBelow,

    /// Check for similarity of the names. Names are deemed similar if the
    /// similarity measurement (the result of heuristic) is **above** the
    /// configured threshold.
    SimilarAbove
  };

public:
  static constexpr std::size_t SmallVectorSize = 8;
  static constexpr std::size_t HeuristicCount =
      static_cast<std::size_t>(Heuristic::Dice) + 1;

  SuspiciousCallArgumentCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const std::size_t MinimumIdentifierNameLength;

  /// The configuration for which heuristics were enabled.
  SmallVector<Heuristic, HeuristicCount> AppliedHeuristics;

  /// The lower and upper bounds for each heuristic, as configured by the user.
  SmallVector<std::pair<int8_t, int8_t>, HeuristicCount> ConfiguredBounds;

  /// The abbreviation-to-abbreviated map for the Abbreviation heuristic.
  llvm::StringMap<std::string> AbbreviationDictionary;

  bool isHeuristicEnabled(Heuristic H) const;
  std::optional<int8_t> getBound(Heuristic H, BoundKind BK) const;

  // Runtime information of the currently analyzed function call.
  SmallVector<QualType, SmallVectorSize> ArgTypes;
  SmallVector<StringRef, SmallVectorSize> ArgNames;
  SmallVector<QualType, SmallVectorSize> ParamTypes;
  SmallVector<StringRef, SmallVectorSize> ParamNames;

  void setParamNamesAndTypes(const FunctionDecl *CalleeFuncDecl);

  void setArgNamesAndTypes(const CallExpr *MatchedCallExpr,
                           std::size_t InitialArgIndex);

  bool areParamAndArgComparable(std::size_t Position1, std::size_t Position2,
                                const ASTContext &Ctx) const;

  bool areArgsSwapped(std::size_t Position1, std::size_t Position2) const;

  bool areNamesSimilar(StringRef Arg, StringRef Param, Heuristic H,
                       BoundKind BK) const;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_SUSPICIOUSCALLARGUMENTCHECK_H
