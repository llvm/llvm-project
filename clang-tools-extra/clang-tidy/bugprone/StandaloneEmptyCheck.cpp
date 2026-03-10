//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StandaloneEmptyCheck.h"
#include "../utils/Matchers.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"
#include "clang/Sema/HeuristicResolver.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void StandaloneEmptyCheck::registerMatchers(MatchFinder *Finder) {
  // Ignore empty calls in a template definition which fall under callExpr
  // non-member matcher even if they are methods.
  const auto NonMemberMatcher =
      expr(ignoringParenImpCasts(
               callExpr(unless(hasAncestor(classTemplateDecl())),
                        callee(functionDecl(hasName("empty"),
                                            unless(returns(voidType())))))
                   .bind("empty")),
           matchers::isDiscarded());
  const auto MemberMatcher =
      expr(ignoringParenImpCasts(cxxMemberCallExpr(callee(
               cxxMethodDecl(hasName("empty"), unless(returns(voidType())))))),
           matchers::isDiscarded())
          .bind("empty");

  Finder->addMatcher(MemberMatcher, this);
  Finder->addMatcher(NonMemberMatcher, this);
}

void StandaloneEmptyCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *MemberCall =
          Result.Nodes.getNodeAs<CXXMemberCallExpr>("empty")) {
    const SourceLocation MemberLoc = MemberCall->getBeginLoc();
    const SourceLocation ReplacementLoc = MemberCall->getExprLoc();
    const SourceRange ReplacementRange =
        SourceRange(ReplacementLoc, ReplacementLoc);

    ASTContext &Context = MemberCall->getRecordDecl()->getASTContext();
    const DeclarationName Name =
        Context.DeclarationNames.getIdentifier(&Context.Idents.get("clear"));

    auto Candidates = HeuristicResolver(Context).lookupDependentName(
        MemberCall->getRecordDecl(), Name, [](const NamedDecl *ND) {
          return isa<CXXMethodDecl>(ND) &&
                 llvm::cast<CXXMethodDecl>(ND)->getMinRequiredArguments() ==
                     0 &&
                 !llvm::cast<CXXMethodDecl>(ND)->isConst();
        });

    const bool HasClear = !Candidates.empty();
    if (HasClear) {
      const auto *Clear = llvm::cast<CXXMethodDecl>(Candidates.at(0));
      const QualType RangeType =
          MemberCall->getImplicitObjectArgument()->getType();
      const bool QualifierIncompatible =
          (!Clear->isVolatile() && RangeType.isVolatileQualified()) ||
          RangeType.isConstQualified();
      if (!QualifierIncompatible) {
        diag(MemberLoc,
             "ignoring the result of 'empty()'; did you mean 'clear()'? ")
            << FixItHint::CreateReplacement(ReplacementRange, "clear");
        return;
      }
    }

    diag(MemberLoc, "ignoring the result of 'empty()'");

  } else if (const auto *NonMemberCall =
                 Result.Nodes.getNodeAs<CallExpr>("empty")) {
    if (NonMemberCall->getNumArgs() != 1)
      return;

    const SourceLocation NonMemberLoc = NonMemberCall->getExprLoc();
    const SourceLocation NonMemberEndLoc = NonMemberCall->getEndLoc();

    const Expr *Arg = NonMemberCall->getArg(0);
    CXXRecordDecl *ArgRecordDecl = Arg->getType()->getAsCXXRecordDecl();
    if (ArgRecordDecl == nullptr)
      return;

    ASTContext &Context = ArgRecordDecl->getASTContext();
    const DeclarationName Name =
        Context.DeclarationNames.getIdentifier(&Context.Idents.get("clear"));

    auto Candidates = HeuristicResolver(Context).lookupDependentName(
        ArgRecordDecl, Name, [](const NamedDecl *ND) {
          return isa<CXXMethodDecl>(ND) &&
                 llvm::cast<CXXMethodDecl>(ND)->getMinRequiredArguments() ==
                     0 &&
                 !llvm::cast<CXXMethodDecl>(ND)->isConst();
        });

    const bool HasClear = !Candidates.empty();

    if (HasClear) {
      const auto *Clear = llvm::cast<CXXMethodDecl>(Candidates.at(0));
      const bool QualifierIncompatible =
          (!Clear->isVolatile() && Arg->getType().isVolatileQualified()) ||
          Arg->getType().isConstQualified();
      if (!QualifierIncompatible) {
        const std::string ReplacementText =
            std::string(Lexer::getSourceText(
                CharSourceRange::getTokenRange(Arg->getSourceRange()),
                *Result.SourceManager, getLangOpts())) +
            ".clear()";
        const SourceRange ReplacementRange =
            SourceRange(NonMemberLoc, NonMemberEndLoc);
        diag(NonMemberLoc,
             "ignoring the result of '%0'; did you mean 'clear()'?")
            << llvm::dyn_cast<NamedDecl>(NonMemberCall->getCalleeDecl())
                   ->getQualifiedNameAsString()
            << FixItHint::CreateReplacement(ReplacementRange, ReplacementText);
        return;
      }
    }

    diag(NonMemberLoc, "ignoring the result of '%0'")
        << llvm::dyn_cast<NamedDecl>(NonMemberCall->getCalleeDecl())
               ->getQualifiedNameAsString();
  }
}

} // namespace clang::tidy::bugprone
