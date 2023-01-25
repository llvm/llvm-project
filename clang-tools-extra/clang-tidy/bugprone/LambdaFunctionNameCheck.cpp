//===--- LambdaFunctionNameCheck.cpp - clang-tidy--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LambdaFunctionNameCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

// Keep track of macro expansions that contain both __FILE__ and __LINE__. If
// such a macro also uses __func__ or __FUNCTION__, we don't want to issue a
// warning because __FILE__ and __LINE__ may be useful even if __func__ or
// __FUNCTION__ is not, especially if the macro could be used in the context of
// either a function body or a lambda body.
class MacroExpansionsWithFileAndLine : public PPCallbacks {
public:
  explicit MacroExpansionsWithFileAndLine(
      LambdaFunctionNameCheck::SourceRangeSet *SME)
      : SuppressMacroExpansions(SME) {}

  void MacroExpands(const Token &MacroNameTok,
                    const MacroDefinition &MD, SourceRange Range,
                    const MacroArgs *Args) override {
    bool HasFile = false;
    bool HasLine = false;
    for (const auto& T : MD.getMacroInfo()->tokens()) {
      if (T.is(tok::identifier)) {
        StringRef IdentName = T.getIdentifierInfo()->getName();
        if (IdentName == "__FILE__") {
          HasFile = true;
        } else if (IdentName == "__LINE__") {
          HasLine = true;
        }
      }
    }
    if (HasFile && HasLine) {
      SuppressMacroExpansions->insert(Range);
    }
  }

private:
  LambdaFunctionNameCheck::SourceRangeSet* SuppressMacroExpansions;
};

} // namespace

void LambdaFunctionNameCheck::registerMatchers(MatchFinder *Finder) {
  // Match on PredefinedExprs inside a lambda.
  Finder->addMatcher(predefinedExpr(hasAncestor(lambdaExpr())).bind("E"),
                     this);
}

void LambdaFunctionNameCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(std::make_unique<MacroExpansionsWithFileAndLine>(
      &SuppressMacroExpansions));
}

void LambdaFunctionNameCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *E = Result.Nodes.getNodeAs<PredefinedExpr>("E");
  if (E->getIdentKind() != PredefinedExpr::Func &&
      E->getIdentKind() != PredefinedExpr::Function) {
    // We don't care about other PredefinedExprs.
    return;
  }
  if (E->getLocation().isMacroID()) {
    auto ER =
        Result.SourceManager->getImmediateExpansionRange(E->getLocation());
    if (SuppressMacroExpansions.find(ER.getAsRange()) !=
        SuppressMacroExpansions.end()) {
      // This is a macro expansion for which we should not warn.
      return;
    }
  }
  diag(E->getLocation(),
       "inside a lambda, '%0' expands to the name of the function call "
       "operator; consider capturing the name of the enclosing function "
       "explicitly")
      << PredefinedExpr::getIdentKindName(E->getIdentKind());
}

} // namespace clang::tidy::bugprone
