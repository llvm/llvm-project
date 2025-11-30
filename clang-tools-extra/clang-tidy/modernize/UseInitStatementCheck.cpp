//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseInitStatementCheck.h"
#include "../utils/ASTUtils.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include <algorithm>
#include <cctype>
#include <optional>
#include <vector>

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

namespace {

// Matches CompoundStmt that contains a PrevStmt immediately followed by
// NextStmt
// FIXME: use hasAdjSubstatements, see
// https://github.com/llvm/llvm-project/pull/169965
AST_MATCHER_P2(CompoundStmt, hasAdjacentStmts,
               ast_matchers::internal::Matcher<Stmt>, DeclMatcher,
               ast_matchers::internal::Matcher<Stmt>, StmtMatcher) {
  const auto Statements = Node.body();

  return std::adjacent_find(
             Statements.begin(), Statements.end(),
             [&](const Stmt *PrevStmt, const Stmt *NextStmt) {
               clang::ast_matchers::internal::BoundNodesTreeBuilder PrevBuilder;
               if (!DeclMatcher.matches(*PrevStmt, Finder, &PrevBuilder))
                 return false;

               clang::ast_matchers::internal::BoundNodesTreeBuilder NextBuilder;
               NextBuilder.addMatch(PrevBuilder);
               if (!StmtMatcher.matches(*NextStmt, Finder, &NextBuilder))
                 return false;

               Builder->addMatch(NextBuilder);
               return true;
             }) != Statements.end();
}

} // namespace

void UseInitStatementCheck::registerMatchers(MatchFinder *Finder) {
  // Matchers for classes with destructors
  const auto ClassWithDtorDecl =
      cxxRecordDecl(hasMethod(cxxDestructorDecl().bind("dtorDecl")));
  const auto ClassWithDtorType =
      hasCanonicalType(hasDeclaration(ClassWithDtorDecl));
  const auto ArrayOfClassWithDtor =
      hasType(arrayType(hasElementType(ClassWithDtorType)));
  const auto HasDtor = anyOf(hasType(ClassWithDtorType), ArrayOfClassWithDtor);

  // Matchers for variable declarations
  const auto SingleVarDeclWithDtor =
      varDecl(HasDtor).bind("singleVar");
  const auto SingleVarDecl =
      varDecl().bind("singleVar");
  const auto RefToBoundVarDecl =
      declRefExpr(to(varDecl(equalsBoundNode("singleVar"))));

  // Matchers for declaration statements that precede if/switch
  const auto PrevDeclStmtWithDtor =
      declStmt(forEach(SingleVarDeclWithDtor)).bind("prevDecl");
  const auto PrevDeclStmt =
      declStmt(forEach(SingleVarDecl)).bind("prevDecl");
  const auto PrevDeclStmtMatcher =
      anyOf(PrevDeclStmtWithDtor, PrevDeclStmt);

  // Helper to create a complete matcher for if/switch statements
  const auto MakeCompoundMatcher = [&](const auto &StmtMatcher,
                                       const std::string &StmtName) {
    const auto StmtMatcherWithCondition =
        StmtMatcher(unless(hasInitStatement(anything())),
                    hasCondition(expr().bind("condition")))
            .bind(StmtName);

    // Ensure the variable is not referenced elsewhere in the compound statement
    const auto NoOtherVarRefs = unless(has(stmt(
        unless(equalsBoundNode(StmtName)), hasDescendant(RefToBoundVarDecl))));

    return compoundStmt(
               unless(isInTemplateInstantiation()),
               hasAdjacentStmts(PrevDeclStmtMatcher, StmtMatcherWithCondition),
               NoOtherVarRefs)
        .bind("compound");
  };

  // Register matchers for if and switch statements
  Finder->addMatcher(MakeCompoundMatcher(ifStmt, "ifStmt"), this);
  Finder->addMatcher(MakeCompoundMatcher(switchStmt, "switchStmt"), this);
}

static bool isLastInCompound(const Stmt *S, const CompoundStmt *P) {
  return !P->body_empty() && P->body_back() == S;
}

static std::string extractDeclStmtText(const DeclStmt *PrevDecl,
                                       const SourceManager *SM,
                                       const LangOptions &LangOpts) {
  const SourceRange CuttingRange = PrevDecl->getSourceRange();
  const CharSourceRange DeclCharRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(CuttingRange), *SM, LangOpts);
  const StringRef DeclStmtText =
      DeclCharRange.isInvalid()
          ? ""
          : Lexer::getSourceText(DeclCharRange, *SM, LangOpts);
  return DeclStmtText.empty() ? "" : DeclStmtText.trim().str() + " ";
}

void UseInitStatementCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *If = Result.Nodes.getNodeAs<Stmt>("ifStmt");
  const auto *Switch = Result.Nodes.getNodeAs<Stmt>("switchStmt");
  const auto *Dtor = Result.Nodes.getNodeAs<CXXDestructorDecl>("dtorDecl");
  const auto *PrevDecl = Result.Nodes.getNodeAs<DeclStmt>("prevDecl");
  const auto *Condition = Result.Nodes.getNodeAs<Expr>("condition");
  const auto *Compound = Result.Nodes.getNodeAs<CompoundStmt>("compound");
  const auto *Statement = If ? If : Switch;

  if (!PrevDecl || !Condition || !Compound || !Statement)
    return;

  if (Dtor && !isLastInCompound(Statement, Compound))
    return;

  auto Diag = diag(PrevDecl->getBeginLoc(),
                   "%select{multiple variable|variable %1}0 declaration "
                   "before %select{if|switch}2 statement could be moved into "
                   "%select{if|switch}2 init statement")
              << (llvm::size(PrevDecl->decls()) == 1)
              << llvm::dyn_cast<VarDecl>(*PrevDecl->decl_begin()) << !If;

  const auto NewInitStmtOpt =
      extractDeclStmtText(PrevDecl, Result.SourceManager, getLangOpts());
  const bool CanFix = !NewInitStmtOpt.empty();
  const SourceRange RemovalRange = PrevDecl->getSourceRange();

  if (CanFix)
    Diag << FixItHint::CreateRemoval(RemovalRange)
         << FixItHint::CreateInsertion(Condition->getBeginLoc(),
                                       NewInitStmtOpt);
}

} // namespace clang::tidy::modernize
