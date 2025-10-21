//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LostStdMoveCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

template <typename Node>
static void extractNodesByIdTo(ArrayRef<BoundNodes> Matches, StringRef ID,
                               llvm::SmallPtrSet<const Node *, 16> &Nodes) {
  for (const BoundNodes &Match : Matches)
    Nodes.insert(Match.getNodeAs<Node>(ID));
}

static llvm::SmallPtrSet<const DeclRefExpr *, 16>
allDeclRefExprsHonourLambda(const VarDecl &VarDecl, const Decl &Decl,
                            ASTContext &Context) {
  auto Matches = match(
      decl(forEachDescendant(
          declRefExpr(to(varDecl(equalsNode(&VarDecl))),
                      unless(hasAncestor(lambdaExpr(hasAnyCapture(lambdaCapture(
                          capturesVar(varDecl(equalsNode(&VarDecl)))))))))
              .bind("declRef"))),
      Decl, Context);
  llvm::SmallPtrSet<const DeclRefExpr *, 16> DeclRefs;
  extractNodesByIdTo(Matches, "declRef", DeclRefs);
  return DeclRefs;
}

static llvm::SmallPtrSet<const VarDecl *, 16>
allVarDeclsExprs(const VarDecl &VarDecl, const Decl &Decl,
                 ASTContext &Context) {
  auto Matches = match(
      decl(forEachDescendant(declRefExpr(
          to(varDecl(equalsNode(&VarDecl))),
          hasParent(decl(
              varDecl(hasType(qualType(referenceType()))).bind("varDecl"))),
          unless(hasAncestor(lambdaExpr(hasAnyCapture(
              lambdaCapture(capturesVar(varDecl(equalsNode(&VarDecl))))))))))),
      Decl, Context);
  llvm::SmallPtrSet<const class VarDecl *, 16> VarDecls;
  extractNodesByIdTo(Matches, "varDecl", VarDecls);
  return VarDecls;
}

static const Expr *
getLastVarUsage(const llvm::SmallPtrSet<const DeclRefExpr *, 16> &Exprs) {
  const Expr *LastExpr = nullptr;
  for (const clang::DeclRefExpr *Expr : Exprs) {
    if (!LastExpr)
      LastExpr = Expr;

    if (LastExpr->getBeginLoc() < Expr->getBeginLoc())
      LastExpr = Expr;
  }

  return LastExpr;
}

namespace {

AST_MATCHER(CXXRecordDecl, hasTrivialMoveConstructor) {
  return Node.hasDefinition() && Node.hasTrivialMoveConstructor();
}

AST_MATCHER_P(Expr, ignoreParens, ast_matchers::internal::Matcher<Expr>,
              InnerMatcher) {
  return InnerMatcher.matches(*Node.IgnoreParens(), Finder, Builder);
}

} // namespace

void LostStdMoveCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StrictMode", StrictMode);
}

LostStdMoveCheck::LostStdMoveCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StrictMode(Options.get("StrictMode", false)) {}

void LostStdMoveCheck::registerMatchers(MatchFinder *Finder) {
  auto ReturnParent =
      hasParent(expr(hasParent(cxxConstructExpr(hasParent(returnStmt())))));

  auto OutermostExpr = expr(unless(hasParent(expr())));
  auto LeafStatement = stmt(OutermostExpr);

  Finder->addMatcher(
      cxxConstructExpr(has(expr(has(ignoreParens(
          declRefExpr(
              // not "return x;"
              unless(ReturnParent),
              unless(hasType(namedDecl(hasName("::std::string_view")))),
              // non-trivial type
              hasType(hasCanonicalType(hasDeclaration(cxxRecordDecl()))),
              // non-trivial X(X&&)
              unless(hasType(hasCanonicalType(
                  hasDeclaration(cxxRecordDecl(hasTrivialMoveConstructor()))))),
              // Not in a cycle
              unless(hasAncestor(forStmt())), unless(hasAncestor(doStmt())),
              unless(hasAncestor(whileStmt())),
              // Not in a body of lambda
              unless(hasAncestor(compoundStmt(hasAncestor(lambdaExpr())))),
              // only non-X&
              unless(hasDeclaration(
                  varDecl(hasType(qualType(lValueReferenceType()))))),
              hasAncestor(LeafStatement.bind("leaf_statement")),
              hasDeclaration(varDecl(hasAncestor(functionDecl().bind("func")))
                                 .bind("decl")))
              .bind("use")))))),
      this);
}

void LostStdMoveCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<VarDecl>("decl");
  const auto *MatchedFunc = Result.Nodes.getNodeAs<FunctionDecl>("func");
  const auto *MatchedUse = Result.Nodes.getNodeAs<Expr>("use");
  const auto *MatchedLeafStatement =
      Result.Nodes.getNodeAs<Stmt>("leaf_statement");

  if (!MatchedDecl->hasLocalStorage()) {
    // global or static variable
    return;
  }

  if (!StrictMode) {
    llvm::SmallPtrSet<const VarDecl *, 16> AllVarDecls =
        allVarDeclsExprs(*MatchedDecl, *MatchedFunc, *Result.Context);
    if (!AllVarDecls.empty()) {
      // x is referenced by local var, it may outlive the "use"
      return;
    }
  }

  llvm::SmallPtrSet<const DeclRefExpr *, 16> AllReferences =
      allDeclRefExprsHonourLambda(*MatchedDecl, *MatchedFunc, *Result.Context);
  const Expr *LastUsage = getLastVarUsage(AllReferences);

  if (LastUsage && LastUsage->getBeginLoc() > MatchedUse->getBeginLoc()) {
    // "use" is not the last reference to x
    return;
  }

  if (LastUsage &&
      LastUsage->getSourceRange() != MatchedUse->getSourceRange()) {
    return;
  }

  // Calculate X usage count in the statement
  llvm::SmallPtrSet<const DeclRefExpr *, 16> DeclRefs;

  SmallVector<BoundNodes, 1> Matches = match(
      findAll(
          declRefExpr(to(varDecl(equalsNode(MatchedDecl))),
                      unless(hasAncestor(lambdaExpr(hasAnyCapture(lambdaCapture(
                          capturesVar(varDecl(equalsNode(MatchedDecl))))))))

                          )
              .bind("ref")),
      *MatchedLeafStatement, *Result.Context);
  extractNodesByIdTo(Matches, "ref", DeclRefs);
  if (DeclRefs.size() > 1) {
    // Unspecified order of evaluation, e.g. f(x, x)
    return;
  }

  const SourceManager &Source = Result.Context->getSourceManager();
  const CharSourceRange Range =
      CharSourceRange::getTokenRange(MatchedUse->getSourceRange());
  const StringRef NeedleExprCode =
      Lexer::getSourceText(Range, Source, Result.Context->getLangOpts());

  if (NeedleExprCode == "=") {
    diag(MatchedUse->getBeginLoc(), "could be std::move()")
        << FixItHint::CreateInsertion(MatchedUse->getBeginLoc(),
                                      (MatchedDecl->getName() +
                                       " = std::move(" +
                                       MatchedDecl->getName() + "),")
                                          .str());
  } else {
    diag(MatchedUse->getBeginLoc(), "could be std::move()")
        << FixItHint::CreateReplacement(
               Range, ("std::move(" + NeedleExprCode + ")").str());
  }
}

} // namespace clang::tidy::performance
