#include "UseStdTieCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

void UseStdTieCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(functionDecl(anyOf(hasOverloadedOperatorName("<"),
                                        hasOverloadedOperatorName(">")))
                         .bind("bad-operator"),
                     this);
}

void UseStdTieCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl =
      Result.Nodes.getNodeAs<FunctionDecl>("bad-operator");
  if (!MatchedDecl || !MatchedDecl->hasBody())
    return;

  // 1. SIMPLE SCANNER: Find all 'return' statements inside the function
  auto ReturnMatches =
      match(stmt(findAll(returnStmt(hasReturnValue(expr().bind("ret_val"))))),
            *MatchedDecl->getBody(), *Result.Context);

  std::vector<std::string> TieArgsLhs;
  std::vector<std::string> TieArgsRhs;
  std::string Autos = "";

  // 2. EXTRACT VARIABLES FROM TEXT
  for (const auto &Node : ReturnMatches) {
    const auto *RetVal = Node.getNodeAs<Expr>("ret_val");
    if (!RetVal)
      continue;

    // Get the raw text, e.g., "lhs.n < rhs.n"
    std::string Text =
        Lexer::getSourceText(
            CharSourceRange::getTokenRange(RetVal->getSourceRange()),
            *Result.SourceManager, Result.Context->getLangOpts())
            .str();
    // Find the operator symbol to split the expression
    size_t OpPos = Text.find('<');
    if (OpPos == std::string::npos)
      OpPos = Text.find('>');

    if (OpPos != std::string::npos) {
      std::string LhsSide = Text.substr(0, OpPos);

      // Cleanup: remove spaces and tabs to avoid compilation errors
      std::string CleanLhs = "";
      for (char c : LhsSide)
        if (c != ' ' && c != '\t' && c != '\n')
          CleanLhs += c;

      // If we successfully extracted the left-hand side pattern
      if (CleanLhs.rfind("lhs.", 0) == 0) {
        std::string VarName = CleanLhs.substr(4);

        if (VarName.find("()") != std::string::npos) {
          std::string CleanName = VarName.substr(0, VarName.length() - 2);
          Autos +=
              "    const auto lhs_" + CleanName + " = lhs." + VarName + ";\n";
          Autos +=
              "    const auto rhs_" + CleanName + " = rhs." + VarName + ";\n";
          TieArgsLhs.push_back("lhs_" + CleanName);
          TieArgsRhs.push_back("rhs_" + CleanName);
        } else {
          TieArgsLhs.push_back("lhs." + VarName);
          TieArgsRhs.push_back("rhs." + VarName);
        }
      }
    }
  }

  // 3. GOLDEN RULE: If no variables were extracted, abort before emitting the
  // warning
  if (TieArgsLhs.empty())
    return;

  // 4. Emit the warning now, since we are sure we have a valid fix
  auto Diag = diag(MatchedDecl->getLocation(),
                   "use std::tie to implement lexicographical comparison");

  // 5. Assemble the dynamic variables
  std::string LhsTuple = TieArgsLhs[0];
  std::string RhsTuple = TieArgsRhs[0];
  for (size_t i = 1; i < TieArgsLhs.size(); ++i) {
    LhsTuple += ", " + TieArgsLhs[i];
    RhsTuple += ", " + TieArgsRhs[i];
  }

  std::string Symbol =
      (MatchedDecl->getNameAsString() == "operator<") ? "<" : ">";

  // 6. Inject the replacement code
  std::string Replacement = "{\n" + Autos + "    return std::tie(" + LhsTuple +
                            ") " + Symbol + " std::tie(" + RhsTuple +
                            ");\n"
                            "}";

  Diag << FixItHint::CreateReplacement(MatchedDecl->getBody()->getSourceRange(),
                                       Replacement);
}

} // namespace clang::tidy::modernize
