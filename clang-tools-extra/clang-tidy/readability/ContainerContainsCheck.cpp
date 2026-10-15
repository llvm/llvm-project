//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ContainerContainsCheck.h"
#include "../utils/ASTUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

// Returns true if the macro body uses the parameter expanded at `ParamUseLoc`
// only parenthesized or as the entire body, so every expansion parses alike.
static bool isParameterUsedOnlyParenthesized(SourceLocation ParamUseLoc,
                                             Preprocessor &PP,
                                             const SourceManager &SM) {
  const StringRef MacroName =
      Lexer::getImmediateMacroName(ParamUseLoc, SM, PP.getLangOpts());
  if (MacroName.empty())
    return false;
  const MacroInfo *MI =
      PP.getMacroDefinitionAtLoc(PP.getIdentifierInfo(MacroName),
                                 SM.getExpansionLoc(ParamUseLoc))
          .getMacroInfo();
  if (!MI)
    return false;

  const ArrayRef<Token> Tokens = MI->tokens();
  const SourceLocation ParamDefLoc = SM.getSpellingLoc(ParamUseLoc);
  const auto *ParamTok = llvm::find_if(Tokens, [ParamDefLoc](const Token &Tok) {
    return Tok.getLocation() == ParamDefLoc;
  });
  if (ParamTok == Tokens.end())
    return false;
  if (Tokens.size() == 1)
    return true;

  const IdentifierInfo *Param = ParamTok->getIdentifierInfo();
  for (size_t I = 0, N = Tokens.size(); I < N; ++I) {
    if (Tokens[I].getIdentifierInfo() != Param)
      continue;
    if (I == 0 || I + 1 == N || Tokens[I - 1].isNot(tok::l_paren) ||
        Tokens[I + 1].isNot(tok::r_paren))
      return false;
    // Parentheses that may complete a call or a macro invocation, including
    // one reached through a parameter or another expansion, do not group.
    if (I >= 2 &&
        Tokens[I - 2].isOneOf(tok::identifier, tok::r_paren, tok::r_square,
                              tok::r_brace, tok::greater, tok::greatergreater))
      return false;
  }
  return true;
}

// Returns true if every macro argument that `Loc` is spelled through satisfies
// `isParameterUsedOnlyParenthesized`.
static bool isSpelledThroughParenthesizedMacroArguments(
    SourceLocation Loc, Preprocessor &PP, const SourceManager &SM) {
  for (; Loc.isMacroID(); Loc = SM.getImmediateMacroCallerLoc(Loc)) {
    SourceLocation ParamUseLoc;
    if (SM.isMacroArgExpansion(Loc, &ParamUseLoc) &&
        !isParameterUsedOnlyParenthesized(ParamUseLoc, PP, SM))
      return false;
  }
  return true;
}

void ContainerContainsCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP,
    Preprocessor * /*ModuleExpanderPP*/) {
  this->PP = PP;
}

void ContainerContainsCheck::registerMatchers(MatchFinder *Finder) {
  const auto Literal0 = integerLiteral(equals(0));
  const auto Literal1 = integerLiteral(equals(1));

  const auto ClassWithContains = cxxRecordDecl(
      hasMethod(cxxMethodDecl(isConst(), parameterCountIs(1), isPublic(),
                              unless(isDeleted()), returns(booleanType()),
                              hasAnyName("contains", "Contains"))
                    .bind("contains_fun")));

  const auto CountCall =
      cxxMemberCallExpr(argumentCountIs(1),
                        callee(cxxMethodDecl(hasAnyName("count", "Count"),
                                             ofClass(ClassWithContains))))
          .bind("call");

  const auto FindCall =
      // Either one argument, or assume the second argument is the position to
      // start searching from.
      cxxMemberCallExpr(
          anyOf(argumentCountIs(1),
                allOf(argumentCountIs(2), hasArgument(1, Literal0))),
          callee(cxxMethodDecl(hasAnyName("find", "Find"),
                               ofClass(ClassWithContains))))
          .bind("call");

  const auto EndCall = cxxMemberCallExpr(
      argumentCountIs(0), callee(cxxMethodDecl(hasAnyName("end", "End"),
                                               ofClass(ClassWithContains))));

  const auto StringNpos = anyOf(declRefExpr(to(varDecl(hasName("npos")))),
                                memberExpr(member(hasName("npos"))));

  Finder->addMatcher(
      traverse(TK_AsIs,
               implicitCastExpr(hasImplicitDestinationType(booleanType()),
                                hasSourceExpression(CountCall))
                   .bind("positiveComparison")),
      this);

  const auto PositiveComparison =
      anyOf(allOf(hasOperatorName("!="), hasOperands(CountCall, Literal0)),
            allOf(hasLHS(CountCall), hasOperatorName(">"), hasRHS(Literal0)),
            allOf(hasLHS(Literal0), hasOperatorName("<"), hasRHS(CountCall)),
            allOf(hasLHS(CountCall), hasOperatorName(">="), hasRHS(Literal1)),
            allOf(hasLHS(Literal1), hasOperatorName("<="), hasRHS(CountCall)),
            allOf(hasOperatorName("!="),
                  hasOperands(FindCall, anyOf(EndCall, StringNpos))));

  const auto NegativeComparison =
      anyOf(allOf(hasOperatorName("=="), hasOperands(CountCall, Literal0)),
            allOf(hasLHS(CountCall), hasOperatorName("<="), hasRHS(Literal0)),
            allOf(hasLHS(Literal0), hasOperatorName(">="), hasRHS(CountCall)),
            allOf(hasLHS(CountCall), hasOperatorName("<"), hasRHS(Literal1)),
            allOf(hasLHS(Literal1), hasOperatorName(">"), hasRHS(CountCall)),
            allOf(hasOperatorName("=="),
                  hasOperands(FindCall, anyOf(EndCall, StringNpos))));

  Finder->addMatcher(
      binaryOperation(
          anyOf(allOf(PositiveComparison, expr().bind("positiveComparison")),
                allOf(NegativeComparison, expr().bind("negativeComparison")))),
      this);
}

void ContainerContainsCheck::check(const MatchFinder::MatchResult &Result) {
  // Extract the information about the match
  const auto *Call = Result.Nodes.getNodeAs<CXXMemberCallExpr>("call");
  const auto *PositiveComparison =
      Result.Nodes.getNodeAs<Expr>("positiveComparison");
  const auto *NegativeComparison =
      Result.Nodes.getNodeAs<Expr>("negativeComparison");
  assert((!PositiveComparison || !NegativeComparison) &&
         "only one of PositiveComparison or NegativeComparison should be set");
  const bool Negated = NegativeComparison != nullptr;
  const auto *Comparison = Negated ? NegativeComparison : PositiveComparison;
  const StringRef ContainsFunName =
      Result.Nodes.getNodeAs<CXXMethodDecl>("contains_fun")->getName();

  // Diagnose the issue.
  const auto Diag = diag(Call->getExprLoc(), "use '%0' to check for membership")
                    << ContainsFunName;

  const SourceLocation ComparisonEndLoc = Comparison->getEndLoc();
  if (ComparisonEndLoc.isInvalid())
    return;

  const SourceManager &SM = *Result.SourceManager;
  const LangOptions &LangOpts = Result.Context->getLangOpts();

  const bool HasPrefix = Comparison->getBeginLoc() != Call->getBeginLoc();
  CharSourceRange PrefixRange = CharSourceRange::getCharRange(
      Comparison->getBeginLoc(), Call->getBeginLoc());
  CharSourceRange FunctionNameRange =
      CharSourceRange::getTokenRange(Call->getExprLoc());
  CharSourceRange ArgumentsAndSuffixRange = CharSourceRange::getTokenRange(
      Call->getArg(0)->getBeginLoc(), ComparisonEndLoc);
  SourceLocation NegationLoc = Call->getBeginLoc();

  if (ComparisonEndLoc.isMacroID()) {
    if (!PP || isa<ImplicitCastExpr>(Comparison) ||
        !Comparison->getType()->isBooleanType() ||
        !utils::rangeIsEntirelyWithinMacroArgument(Comparison->getSourceRange(),
                                                   &SM) ||
        !isSpelledThroughParenthesizedMacroArguments(Comparison->getBeginLoc(),
                                                     *PP, SM) ||
        !isSpelledThroughParenthesizedMacroArguments(ComparisonEndLoc, *PP, SM))
      return;

    const CharSourceRange CallRange = Lexer::makeFileCharRange(
        CharSourceRange::getTokenRange(Call->getSourceRange()), SM, LangOpts);
    if (HasPrefix)
      PrefixRange = Lexer::makeFileCharRange(PrefixRange, SM, LangOpts);
    FunctionNameRange =
        Lexer::makeFileCharRange(FunctionNameRange, SM, LangOpts);
    ArgumentsAndSuffixRange =
        Lexer::makeFileCharRange(ArgumentsAndSuffixRange, SM, LangOpts);
    if (CallRange.isInvalid() || PrefixRange.isInvalid() ||
        FunctionNameRange.isInvalid() || ArgumentsAndSuffixRange.isInvalid())
      return;
    NegationLoc = CallRange.getBegin();
  }

  bool Invalid = false;
  const StringRef SearchExprText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(Call->getArg(0)->getSourceRange()), SM,
      LangOpts, &Invalid);
  if (Invalid)
    return;

  // Remove everything before the function call.
  if (HasPrefix)
    Diag << FixItHint::CreateRemoval(PrefixRange);

  // Rename the function to `contains`.
  Diag << FixItHint::CreateReplacement(FunctionNameRange, ContainsFunName);

  // Replace arguments and everything after the function call.
  Diag << FixItHint::CreateReplacement(ArgumentsAndSuffixRange,
                                       (SearchExprText + ")").str());

  // Add negation if necessary.
  if (Negated)
    Diag << FixItHint::CreateInsertion(NegationLoc, "!");
}

} // namespace clang::tidy::readability
