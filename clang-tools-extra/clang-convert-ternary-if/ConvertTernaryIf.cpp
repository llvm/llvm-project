//===--- ConvertTernaryIf.cpp ---------------------------------------------===//
//
// Implements a tool that refactors between ternary (?:) expressions
// and equivalent if/else statements.
//
// Usage Example:
//   clang-convert-ternary-if test.cpp --
//
//===----------------------------------------------------------------------===//

#include "ConvertTernaryIf.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::ast_matchers;

namespace clang {
namespace convertternary {


// Callback: Called when a match is found
void ConvertTernaryIfCallback::run(const MatchFinder::MatchResult &Result) {
  //Initialize the Rewriter safely (fixes segmentation fault)
  if (!IsInitialized) {
    if (Result.SourceManager && Result.Context) {
      TheRewriter.setSourceMgr(*Result.SourceManager,
                               Result.Context->getLangOpts());
      IsInitialized = true;
      llvm::errs() << "Rewriter initialized successfully.\n";
    } else {
      llvm::errs() << "Error: Missing SourceManager or Context.\n";
      return;
    }
  }

  const auto *CondOp = Result.Nodes.getNodeAs<ConditionalOperator>("condOp");
  const auto *IfStmtNode = Result.Nodes.getNodeAs<IfStmt>("ifStmt");

  const SourceManager &SM = *Result.SourceManager;

  // === Convert Ternary -> If ===
  if (CondOp) {
    const Expr *Cond = CondOp->getCond();
    const Expr *TrueExpr = CondOp->getTrueExpr();
    const Expr *FalseExpr = CondOp->getFalseExpr();

    auto getText = [&](const Expr *E) -> std::string {
      return Lexer::getSourceText(CharSourceRange::getTokenRange(E->getSourceRange()), SM,
                                  Result.Context->getLangOpts())
          .str();
    };

    std::string CondText = getText(Cond);
    std::string TrueText = getText(TrueExpr);
    std::string FalseText = getText(FalseExpr);

    std::string IfReplacement = "if (" + CondText + ") {\n  " + TrueText +
                                ";\n} else {\n  " + FalseText + ";\n}";

    TheRewriter.ReplaceText(CondOp->getSourceRange(), IfReplacement);
    llvm::errs() << "Converted ternary to if/else.\n";
  }

  // === Convert If -> Ternary ===
  if (IfStmtNode) {
    const Expr *Cond = IfStmtNode->getCond();
    const Stmt *Then = IfStmtNode->getThen();
    const Stmt *Else = IfStmtNode->getElse();

    if (!Then || !Else)
      return;

    auto getTextStmt = [&](const Stmt *S) -> std::string {
      return Lexer::getSourceText(CharSourceRange::getTokenRange(S->getSourceRange()), SM,
                                  Result.Context->getLangOpts())
          .str();
    };

    std::string CondText = Lexer::getSourceText(
                               CharSourceRange::getTokenRange(Cond->getSourceRange()), SM,
                               Result.Context->getLangOpts())
                               .str();

    std::string ThenText = getTextStmt(Then);
    std::string ElseText = getTextStmt(Else);

    std::string Ternary =
        "(" + CondText + ") ? " + ThenText + " : " + ElseText + ";";

    TheRewriter.ReplaceText(IfStmtNode->getSourceRange(), Ternary);
    llvm::errs() << "Converted if/else to ternary.\n";
  }
}

// === Register AST Matchers ===
void setupMatchers(MatchFinder &Finder, ConvertTernaryIfCallback &Callback) {
  Finder.addMatcher(
      conditionalOperator(isExpansionInMainFile()).bind("condOp"), &Callback);

  Finder.addMatcher(
      ifStmt(hasThen(stmt()), hasElse(stmt()), isExpansionInMainFile())
          .bind("ifStmt"),
      &Callback);
}

} // namespace convertternary
} // namespace clang

