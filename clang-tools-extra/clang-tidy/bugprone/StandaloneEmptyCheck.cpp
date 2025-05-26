//===--- StandaloneEmptyCheck.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StandaloneEmptyCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Lexer.h"
#include "clang/Sema/HeuristicResolver.h"
#include "llvm/Support/Casting.h"

namespace clang::tidy::bugprone {

using ast_matchers::BoundNodes;
using ast_matchers::callee;
using ast_matchers::callExpr;
using ast_matchers::classTemplateDecl;
using ast_matchers::cxxMemberCallExpr;
using ast_matchers::cxxMethodDecl;
using ast_matchers::expr;
using ast_matchers::functionDecl;
using ast_matchers::hasAncestor;
using ast_matchers::hasName;
using ast_matchers::hasParent;
using ast_matchers::ignoringImplicit;
using ast_matchers::ignoringParenImpCasts;
using ast_matchers::MatchFinder;
using ast_matchers::optionally;
using ast_matchers::returns;
using ast_matchers::stmt;
using ast_matchers::stmtExpr;
using ast_matchers::unless;
using ast_matchers::voidType;

const Expr *getCondition(const BoundNodes &Nodes, const StringRef NodeId) {
  const auto *If = Nodes.getNodeAs<IfStmt>(NodeId);
  if (If != nullptr)
    return If->getCond();

  const auto *For = Nodes.getNodeAs<ForStmt>(NodeId);
  if (For != nullptr)
    return For->getCond();

  const auto *While = Nodes.getNodeAs<WhileStmt>(NodeId);
  if (While != nullptr)
    return While->getCond();

  const auto *Do = Nodes.getNodeAs<DoStmt>(NodeId);
  if (Do != nullptr)
    return Do->getCond();

  const auto *Switch = Nodes.getNodeAs<SwitchStmt>(NodeId);
  if (Switch != nullptr)
    return Switch->getCond();

  return nullptr;
}

void StandaloneEmptyCheck::registerMatchers(ast_matchers::MatchFinder *Finder) {
  // Ignore empty calls in a template definition which fall under callExpr
  // non-member matcher even if they are methods.
  const auto NonMemberMatcher = expr(ignoringImplicit(ignoringParenImpCasts(
      callExpr(
          hasParent(stmt(optionally(hasParent(stmtExpr().bind("stexpr"))))
                        .bind("parent")),
          unless(hasAncestor(classTemplateDecl())),
          callee(functionDecl(hasName("empty"), unless(returns(voidType())))))
          .bind("empty"))));
  const auto MemberMatcher =
      expr(ignoringImplicit(ignoringParenImpCasts(cxxMemberCallExpr(
               hasParent(stmt(optionally(hasParent(stmtExpr().bind("stexpr"))))
                             .bind("parent")),
               callee(cxxMethodDecl(hasName("empty"),
                                    unless(returns(voidType()))))))))
          .bind("empty");

  Finder->addMatcher(MemberMatcher, this);
  Finder->addMatcher(NonMemberMatcher, this);
}

void StandaloneEmptyCheck::check(const MatchFinder::MatchResult &Result) {
  // Skip if the parent node is Expr.
  if (Result.Nodes.getNodeAs<Expr>("parent"))
    return;

  const auto *PParentStmtExpr = Result.Nodes.getNodeAs<Expr>("stexpr");
  const auto *ParentCompStmt = Result.Nodes.getNodeAs<CompoundStmt>("parent");
  const auto *ParentCond = getCondition(Result.Nodes, "parent");
  const auto *ParentReturnStmt = Result.Nodes.getNodeAs<ReturnStmt>("parent");

  if (const auto *MemberCall =
          Result.Nodes.getNodeAs<CXXMemberCallExpr>("empty")) {
    // Skip if it's a condition of the parent statement.
    if (ParentCond == MemberCall->getExprStmt())
      return;
    // Skip if it's the last statement in the GNU extension
    // statement expression.
    if (PParentStmtExpr && ParentCompStmt &&
        ParentCompStmt->body_back() == MemberCall->getExprStmt())
      return;
    // Skip if it's a return statement
    if (ParentReturnStmt)
      return;

    SourceLocation MemberLoc = MemberCall->getBeginLoc();
    SourceLocation ReplacementLoc = MemberCall->getExprLoc();
    SourceRange ReplacementRange = SourceRange(ReplacementLoc, ReplacementLoc);

    ASTContext &Context = MemberCall->getRecordDecl()->getASTContext();
    DeclarationName Name =
        Context.DeclarationNames.getIdentifier(&Context.Idents.get("clear"));

    auto Candidates = HeuristicResolver(Context).lookupDependentName(
        MemberCall->getRecordDecl(), Name, [](const NamedDecl *ND) {
          return isa<CXXMethodDecl>(ND) &&
                 llvm::cast<CXXMethodDecl>(ND)->getMinRequiredArguments() ==
                     0 &&
                 !llvm::cast<CXXMethodDecl>(ND)->isConst();
        });

    bool HasClear = !Candidates.empty();
    if (HasClear) {
      const auto *Clear = llvm::cast<CXXMethodDecl>(Candidates.at(0));
      QualType RangeType = MemberCall->getImplicitObjectArgument()->getType();
      bool QualifierIncompatible =
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
    if (ParentCond == NonMemberCall->getExprStmt())
      return;
    if (PParentStmtExpr && ParentCompStmt &&
        ParentCompStmt->body_back() == NonMemberCall->getExprStmt())
      return;
    if (ParentReturnStmt)
      return;
    if (NonMemberCall->getNumArgs() != 1)
      return;

    SourceLocation NonMemberLoc = NonMemberCall->getExprLoc();
    SourceLocation NonMemberEndLoc = NonMemberCall->getEndLoc();

    const Expr *Arg = NonMemberCall->getArg(0);
    CXXRecordDecl *ArgRecordDecl = Arg->getType()->getAsCXXRecordDecl();
    if (ArgRecordDecl == nullptr)
      return;

    ASTContext &Context = ArgRecordDecl->getASTContext();
    DeclarationName Name =
        Context.DeclarationNames.getIdentifier(&Context.Idents.get("clear"));

    auto Candidates = HeuristicResolver(Context).lookupDependentName(
        ArgRecordDecl, Name, [](const NamedDecl *ND) {
          return isa<CXXMethodDecl>(ND) &&
                 llvm::cast<CXXMethodDecl>(ND)->getMinRequiredArguments() ==
                     0 &&
                 !llvm::cast<CXXMethodDecl>(ND)->isConst();
        });

    bool HasClear = !Candidates.empty();

    if (HasClear) {
      const auto *Clear = llvm::cast<CXXMethodDecl>(Candidates.at(0));
      bool QualifierIncompatible =
          (!Clear->isVolatile() && Arg->getType().isVolatileQualified()) ||
          Arg->getType().isConstQualified();
      if (!QualifierIncompatible) {
        std::string ReplacementText =
            std::string(Lexer::getSourceText(
                CharSourceRange::getTokenRange(Arg->getSourceRange()),
                *Result.SourceManager, getLangOpts())) +
            ".clear()";
        SourceRange ReplacementRange =
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
