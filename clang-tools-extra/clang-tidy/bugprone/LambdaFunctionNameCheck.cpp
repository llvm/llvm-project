//===--- LambdaFunctionNameCheck.cpp - clang-tidy--------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LambdaFunctionNameCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

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
    bool has_file = false;
    bool has_line = false;
    for (const auto& T : MD.getMacroInfo()->tokens()) {
      if (T.is(tok::identifier)) {
        StringRef IdentName = T.getIdentifierInfo()->getName();
        if (IdentName == "__FILE__") {
          has_file = true;
        } else if (IdentName == "__LINE__") {
          has_line = true;
        }
      }
    }
    if (has_file && has_line) {
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

void LambdaFunctionNameCheck::registerPPCallbacks(CompilerInstance &Compiler) {
  Compiler.getPreprocessor().addPPCallbacks(
      llvm::make_unique<MacroExpansionsWithFileAndLine>(
          &SuppressMacroExpansions));
}

void LambdaFunctionNameCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *E = Result.Nodes.getNodeAs<PredefinedExpr>("E");
  if (E->getIdentType() != PredefinedExpr::Func &&
      E->getIdentType() != PredefinedExpr::Function) {
    // We don't care about other PredefinedExprs.
    return;
  }
  if (E->getLocation().isMacroID()) {
    auto ER =
        Result.SourceManager->getImmediateExpansionRange(E->getLocation());
    if (SuppressMacroExpansions.find(SourceRange(ER.first, ER.second)) !=
        SuppressMacroExpansions.end()) {
      // This is a macro expansion for which we should not warn.
      return;
    }
  }
  diag(E->getLocation(),
       "inside a lambda, '%0' expands to the name of the function call "
       "operator; consider capturing the name of the enclosing function "
       "explicitly")
      << PredefinedExpr::getIdentTypeName(E->getIdentType());
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
