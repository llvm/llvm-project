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

  // 1. ESCÁNER SIMPLE: Buscamos todos los 'return' dentro de la función
  auto ReturnMatches =
      match(stmt(findAll(returnStmt(hasReturnValue(expr().bind("ret_val"))))),
            *MatchedDecl->getBody(), *Result.Context);

  std::vector<std::string> TieArgsLhs;
  std::vector<std::string> TieArgsRhs;
  std::string Autos = "";

  // 2. EXTRAER VARIABLES DEL TEXTO
  for (const auto &Node : ReturnMatches) {
    const auto *RetVal = Node.getNodeAs<Expr>("ret_val");
    if (!RetVal)
      continue;

    // Obtenemos el texto en bruto, ej: "lhs.n < rhs.n"
    std::string Text =
        Lexer::getSourceText(
            CharSourceRange::getTokenRange(RetVal->getSourceRange()),
            *Result.SourceManager, Result.Context->getLangOpts())
            .str();

    // Buscamos dónde está el símbolo para partir la frase
    size_t OpPos = Text.find('<');
    if (OpPos == std::string::npos)
      OpPos = Text.find('>');

    if (OpPos != std::string::npos) {
      std::string LhsSide = Text.substr(0, OpPos);

      // Limpieza extrema: quitamos espacios y tabulaciones para que no rompa la
      // compilación
      std::string CleanLhs = "";
      for (char c : LhsSide)
        if (c != ' ' && c != '\t' && c != '\n')
          CleanLhs += c;

      // Si nos ha quedado "lhs.n" o "lhs.d()"
      if (CleanLhs.rfind("lhs.", 0) == 0) {
        std::string VarName = CleanLhs.substr(4); // Extrae "n", "s" o "d()"

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

  // 3. LA REGLA DE ORO: Si no hemos extraído nada, abortamos ANTES de lanzar la
  // advertencia
  if (TieArgsLhs.empty())
    return;

  // 4. Lanzamos la advertencia AHORA, porque sabemos que tenemos el parche
  // asegurado
  auto Diag = diag(MatchedDecl->getLocation(),
                   "use std::tie to implement lexicographical comparison");

  // 5. Ensamblamos las variables dinámicas
  std::string LhsTuple = TieArgsLhs[0];
  std::string RhsTuple = TieArgsRhs[0];
  for (size_t i = 1; i < TieArgsLhs.size(); ++i) {
    LhsTuple += ", " + TieArgsLhs[i];
    RhsTuple += ", " + TieArgsRhs[i];
  }

  std::string Symbol =
      (MatchedDecl->getNameAsString() == "operator<") ? "<" : ">";

  // 6. Inyectamos el código perfecto
  std::string Replacement = "{\n" + Autos + "    return std::tie(" + LhsTuple +
                            ") " + Symbol + " std::tie(" + RhsTuple +
                            ");\n"
                            "}";

  Diag << FixItHint::CreateReplacement(MatchedDecl->getBody()->getSourceRange(),
                                       Replacement);
}

} // namespace clang::tidy::modernize
