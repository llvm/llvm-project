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
  const auto ClassWithDtorDecl = cxxRecordDecl(hasMethod(cxxDestructorDecl()));
  const auto ClassWithDtorType =
      hasCanonicalType(hasDeclaration(ClassWithDtorDecl));
  const auto ArrayOfClassWithDtor =
      hasType(arrayType(hasElementType(ClassWithDtorType)));

  const auto SingleVarDecl =
      varDecl(unless(anyOf(hasType(ClassWithDtorType), ArrayOfClassWithDtor)))
          .bind("singleVar");
  const auto RefToBoundVarDecl =
      declRefExpr(to(varDecl(equalsBoundNode("singleVar"))));

  // Matcher for the declaration statement that precedes the if/switch
  const auto PrevDeclStmt = declStmt(forEach(SingleVarDecl)).bind("prevDecl");

  // Matcher for a condition that references the variable from prevDecl
  const auto ConditionWithVarRef =
      expr(forEachDescendant(RefToBoundVarDecl)).bind("condition");

  // Helper to create a complete matcher for if/switch statements
  const auto MakeCompoundMatcher = [&](const auto &StmtMatcher,
                                       const std::string &StmtName) {
    const auto StmtMatcherWithCondition =
        StmtMatcher(unless(hasInitStatement(anything())),
                    hasCondition(ConditionWithVarRef))
            .bind(StmtName);

    // Ensure the variable is not referenced elsewhere in the compound statement
    const auto NoOtherVarRefs = unless(has(stmt(
        unless(equalsBoundNode(StmtName)), hasDescendant(RefToBoundVarDecl))));

    return compoundStmt(
        unless(isInTemplateInstantiation()),
        hasAdjacentStmts(PrevDeclStmt, StmtMatcherWithCondition),
        NoOtherVarRefs);
  };

  Finder->addMatcher(MakeCompoundMatcher(ifStmt, "ifStmt"), this);
  Finder->addMatcher(MakeCompoundMatcher(switchStmt, "switchStmt"), this);
}

static StringRef normDeclStmtText(StringRef Text) {
  Text = Text.trim();
  while (Text.consume_back(";")) {
    Text = Text.rtrim();
  }
  return Text;
}

void UseInitStatementCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *If = Result.Nodes.getNodeAs<IfStmt>("ifStmt");
  const auto *Switch = Result.Nodes.getNodeAs<SwitchStmt>("switchStmt");
  const auto *PrevDecl = Result.Nodes.getNodeAs<DeclStmt>("prevDecl");
  const auto *Condition = Result.Nodes.getNodeAs<Expr>("condition");
  const Stmt *Statement = If ? static_cast<const Stmt *>(If) : Switch;

  if (!PrevDecl || !Condition || !Statement)
    return;

  const SourceRange RemovalRange = PrevDecl->getSourceRange();
  const bool CanFix =
      utils::rangeCanBeFixed(RemovalRange, Result.SourceManager) &&
      !Condition->getBeginLoc().isMacroID();

  const StringRef DeclStmtText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(PrevDecl->getSourceRange()),
      *Result.SourceManager, getLangOpts());
  const auto NewInitStmt = normDeclStmtText(DeclStmtText).str() + "; ";

  auto Diag = diag(PrevDecl->getBeginLoc(),
                   "%select{multiple variable|variable %1}0 declaration "
                   "before %select{if|switch}2 statement could be moved into "
                   "%select{if|switch}2 init statement")
              << (llvm::size(PrevDecl->decls()) == 1)
              << llvm::dyn_cast<VarDecl>(*PrevDecl->decl_begin()) << !If;
  if (CanFix)
    Diag << FixItHint::CreateRemoval(RemovalRange)
         << FixItHint::CreateInsertion(Condition->getBeginLoc(), NewInitStmt);
}

} // namespace clang::tidy::modernize
