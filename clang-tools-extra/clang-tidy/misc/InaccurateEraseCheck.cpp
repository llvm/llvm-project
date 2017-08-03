//===--- InaccurateEraseCheck.cpp - clang-tidy-----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InaccurateEraseCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

namespace {
AST_MATCHER(Decl, isInStdNamespace) { return Node.isInStdNamespace(); }
}

void InaccurateEraseCheck::registerMatchers(MatchFinder *Finder) {
  // Only register the matchers for C++; the functionality currently does not
  // provide any benefit to other languages, despite being benign.
  if (!getLangOpts().CPlusPlus)
    return;

  const auto EndCall =
      callExpr(
          callee(functionDecl(hasAnyName("remove", "remove_if", "unique"))),
          hasArgument(
              1,
              anyOf(cxxConstructExpr(has(ignoringImplicit(
                        cxxMemberCallExpr(callee(cxxMethodDecl(hasName("end"))))
                            .bind("end")))),
                    anything())))
          .bind("alg");

  const auto DeclInStd = type(hasUnqualifiedDesugaredType(
      tagType(hasDeclaration(decl(isInStdNamespace())))));
  Finder->addMatcher(
      cxxMemberCallExpr(
          on(anyOf(hasType(DeclInStd), hasType(pointsTo(DeclInStd)))),
          callee(cxxMethodDecl(hasName("erase"))), argumentCountIs(1),
          hasArgument(0, has(ignoringImplicit(
                             anyOf(EndCall, has(ignoringImplicit(EndCall)))))),
          unless(isInTemplateInstantiation()))
          .bind("erase"),
      this);
}

void InaccurateEraseCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MemberCall =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("erase");
  const auto *EndExpr =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("end");
  const SourceLocation Loc = MemberCall->getLocStart();

  FixItHint Hint;

  if (!Loc.isMacroID() && EndExpr) {
    const auto *AlgCall = Result.Nodes.getNodeAs<CallExpr>("alg");
    std::string ReplacementText = Lexer::getSourceText(
        CharSourceRange::getTokenRange(EndExpr->getSourceRange()),
        *Result.SourceManager, getLangOpts());
    const SourceLocation EndLoc = Lexer::getLocForEndOfToken(
        AlgCall->getLocEnd(), 0, *Result.SourceManager, getLangOpts());
    Hint = FixItHint::CreateInsertion(EndLoc, ", " + ReplacementText);
  }

  diag(Loc, "this call will remove at most one item even when multiple items "
            "should be removed")
      << Hint;
}

} // namespace misc
} // namespace tidy
} // namespace clang
