//===--- MagicNumbersCheck.h - clang-tidy-----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_MAGICNUMBERSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_MAGICNUMBERSCHECK_H

#include "../ClangTidy.h"
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/SmallVector.h>
#include <vector>

namespace clang {
namespace tidy {
namespace readability {

/// Detects magic numbers, integer and floating point literals embedded in code.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability-magic-numbers.html
class MagicNumbersCheck : public ClangTidyCheck {
public:
  MagicNumbersCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  bool isConstant(const clang::ast_matchers::MatchFinder::MatchResult &Result,
                  const clang::Expr &ExprResult) const;

  bool isIgnoredValue(const IntegerLiteral *Literal) const;
  bool isIgnoredValue(const FloatingLiteral *Literal) const;

  bool isSyntheticValue(const clang::SourceManager *,
                        const FloatingLiteral *) const {
    return false;
  }

  bool isSyntheticValue(const clang::SourceManager *SourceManager,
                        const IntegerLiteral *Literal) const;

  template <typename L>
  void checkBoundMatch(const ast_matchers::MatchFinder::MatchResult &Result,
                       const char *BoundName) {
    const L *MatchedLiteral = Result.Nodes.getNodeAs<L>(BoundName);
    if (!MatchedLiteral)
      return;

    if (Result.SourceManager->isMacroBodyExpansion(
            MatchedLiteral->getLocation()))
      return;

    if (isIgnoredValue(MatchedLiteral))
      return;

    if (isConstant(Result, *MatchedLiteral))
      return;

    if (isSyntheticValue(Result.SourceManager, MatchedLiteral))
      return;

    const StringRef LiteralSourceText = Lexer::getSourceText(
        CharSourceRange::getTokenRange(MatchedLiteral->getSourceRange()),
        *Result.SourceManager, getLangOpts());

    diag(MatchedLiteral->getLocation(),
         "%0 is a magic number; consider replacing it with a named constant")
        << LiteralSourceText;
  }

  const bool IgnoreAllFloatingPointValues;
  const bool IgnorePowersOf2IntegerValues;

  constexpr static unsigned SensibleNumberOfMagicValueExceptions = 16;

  constexpr static llvm::APFloat::roundingMode DefaultRoundingMode =
      llvm::APFloat::rmNearestTiesToEven;

  llvm::SmallVector<int64_t, SensibleNumberOfMagicValueExceptions>
      IgnoredIntegerValues;
  llvm::SmallVector<float, SensibleNumberOfMagicValueExceptions>
      IgnoredFloatingPointValues;
  llvm::SmallVector<double, SensibleNumberOfMagicValueExceptions>
      IgnoredDoublePointValues;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_MAGICNUMBERSCHECK_H
