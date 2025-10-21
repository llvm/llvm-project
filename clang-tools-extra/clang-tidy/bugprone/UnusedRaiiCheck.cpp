//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnusedRaiiCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {
AST_MATCHER(CXXRecordDecl, hasNonTrivialDestructor) {
  // TODO: If the dtor is there but empty we don't want to warn either.
  return Node.hasDefinition() && Node.hasNonTrivialDestructor();
}
} // namespace

void UnusedRaiiCheck::registerMatchers(MatchFinder *Finder) {
  // Look for temporaries that are constructed in-place and immediately
  // destroyed.
  Finder->addMatcher(
      mapAnyOf(cxxConstructExpr, cxxUnresolvedConstructExpr)
          .with(hasParent(compoundStmt().bind("compound")),
                anyOf(hasType(hasCanonicalType(recordType(hasDeclaration(
                          cxxRecordDecl(hasNonTrivialDestructor()))))),
                      hasType(hasCanonicalType(templateSpecializationType(
                          hasDeclaration(classTemplateDecl(has(
                              cxxRecordDecl(hasNonTrivialDestructor())))))))))
          .bind("expr"),
      this);
}

template <typename T>
static void reportDiagnostic(DiagnosticBuilder D, const T *Node, SourceRange SR,
                             bool DefaultConstruction) {
  const char *Replacement = " give_me_a_name";

  // If this is a default ctor we have to remove the parens or we'll introduce a
  // most vexing parse.
  if (DefaultConstruction) {
    D << FixItHint::CreateReplacement(CharSourceRange::getTokenRange(SR),
                                      Replacement);
    return;
  }

  // Otherwise just suggest adding a name. To find the place to insert the name
  // find the first TypeLoc in the children of E, which always points to the
  // written type.
  D << FixItHint::CreateInsertion(SR.getBegin(), Replacement);
}

void UnusedRaiiCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *E = Result.Nodes.getNodeAs<Expr>("expr");

  // We ignore code expanded from macros to reduce the number of false
  // positives.
  if (E->getBeginLoc().isMacroID())
    return;

  // Don't emit a warning for the last statement in the surrounding compound
  // statement.
  const auto *CS = Result.Nodes.getNodeAs<CompoundStmt>("compound");
  const auto *LastExpr = dyn_cast<Expr>(CS->body_back());

  if (LastExpr && E == LastExpr->IgnoreUnlessSpelledInSource())
    return;

  // Emit a warning.
  auto D = diag(E->getBeginLoc(), "object destroyed immediately after "
                                  "creation; did you mean to name the object?");

  if (const auto *Node = dyn_cast<CXXConstructExpr>(E))
    reportDiagnostic(D, Node, Node->getParenOrBraceRange(),
                     Node->getNumArgs() == 0 ||
                         isa<CXXDefaultArgExpr>(Node->getArg(0)));
  if (const auto *Node = dyn_cast<CXXUnresolvedConstructExpr>(E)) {
    auto SR = SourceRange(Node->getLParenLoc(), Node->getRParenLoc());
    auto DefaultConstruction = Node->getNumArgs() == 0;
    if (!DefaultConstruction) {
      auto *FirstArg = Node->getArg(0);
      DefaultConstruction = isa<CXXDefaultArgExpr>(FirstArg);
      if (auto *ILE = dyn_cast<InitListExpr>(FirstArg)) {
        DefaultConstruction = ILE->getNumInits() == 0;
        SR = SourceRange(ILE->getLBraceLoc(), ILE->getRBraceLoc());
      }
    }
    reportDiagnostic(D, Node, SR, DefaultConstruction);
  }
}

} // namespace clang::tidy::bugprone
