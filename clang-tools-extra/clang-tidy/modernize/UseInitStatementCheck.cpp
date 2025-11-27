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

class VariableUsageVisitor : public RecursiveASTVisitor<VariableUsageVisitor> {
public:
  explicit VariableUsageVisitor(const VarDecl *TargetVar)
      : TargetVar(TargetVar) {}

  bool VisitDeclRefExpr(DeclRefExpr *DRE) {
    assert(!FoundUsage);
    FoundUsage = (dyn_cast<VarDecl>(DRE->getDecl()) == TargetVar);
    return !FoundUsage;
  }

  bool foundUsage() const { return FoundUsage; }

private:
  const VarDecl *TargetVar;
  bool FoundUsage = false;
};

// Matches CompoundStmt that contains a DeclStmt immediately followed by
// a statement matching the inner matcher.
AST_MATCHER_P2(CompoundStmt, hasAdjacentStmts,
               ast_matchers::internal::Matcher<DeclStmt>, DeclMatcher,
               ast_matchers::internal::Matcher<Stmt>, StmtMatcher) {
  const auto Statements = Node.body();

  return std::adjacent_find(
             Statements.begin(), Statements.end(),
             [&](const Stmt *Prev, const Stmt *NextStmt) {
               const auto *PrevDecl = dyn_cast<DeclStmt>(Prev);
               if (!PrevDecl)
                 return false;

               clang::ast_matchers::internal::BoundNodesTreeBuilder DeclBuilder;
               if (!DeclMatcher.matches(*PrevDecl, Finder, &DeclBuilder))
                 return false;

               clang::ast_matchers::internal::BoundNodesTreeBuilder StmtBuilder;
               StmtBuilder.addMatch(DeclBuilder);
               if (!StmtMatcher.matches(*NextStmt, Finder, &StmtBuilder))
                 return false;

               Builder->addMatch(StmtBuilder);
               return true;
             }) != Statements.end();
}

} // namespace

// Collects all VarDecl references from an expression.
static std::vector<const VarDecl *> collectVarDeclsInExpr(const Expr *E) {
  class VarCollector : public RecursiveASTVisitor<VarCollector> {
  public:
    std::vector<const VarDecl *> Vars;

    bool VisitDeclRefExpr(DeclRefExpr *DRE) {
      if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl()))
        Vars.push_back(VD);
      return true; // Continue traversing
    }
  };

  VarCollector Collector;
  Collector.TraverseStmt(const_cast<Expr *>(E));
  return Collector.Vars;
}

// Checks if all variables in DeclStmt are used in the condition.
static bool allVarsUsedInCondition(const DeclStmt *DS, const Expr *Condition) {
  const auto ConditionVars = collectVarDeclsInExpr(Condition);
  return llvm::all_of(DS->decls(), [&](const auto *D) {
    const auto *VD = dyn_cast<VarDecl>(D);
    assert(VD); // TODO:
    return llvm::is_contained(ConditionVars, VD);
  });
}

// Checks if any variable in DeclStmt is used after the statement.
static bool anyVarUsedAfterStmt(const DeclStmt *DS, const Stmt *STMT,
                                const CompoundStmt *ParentCompound) {
  const auto &Statements = ParentCompound->body();
  const auto CurrentPosition = llvm::find(Statements, STMT);
  if (CurrentPosition == Statements.end())
    return false;

  const auto StatementsAfter = std::next(CurrentPosition);
  if (StatementsAfter == Statements.end())
    return false;

  return llvm::any_of(DS->decls(), [&](const auto *D) {
    const auto *VD = dyn_cast<VarDecl>(D);
    assert(VD); // TODO:
    return llvm::any_of(llvm::make_range(StatementsAfter, Statements.end()),
                        [&](const Stmt *S) {
                          VariableUsageVisitor Visitor(VD);
                          Visitor.TraverseStmt(const_cast<Stmt *>(S));
                          return Visitor.foundUsage();
                        });
  });
}

void UseInitStatementCheck::registerMatchers(MatchFinder *Finder) {
  static constexpr auto MakeMatcher = [](const auto &StmtMatcher,
                                         StringRef Name) {
    return compoundStmt(unless(isInTemplateInstantiation()), hasAdjacentStmts(
                            declStmt().bind("prevDecl"),
                            StmtMatcher(unless(hasInitStatement(anything())),
                                        hasCondition(expr().bind("condition")))
                                .bind(Name)))
        .bind("compoundStmt");
  };

  Finder->addMatcher(MakeMatcher(ifStmt, "ifStmt"), this);
  Finder->addMatcher(MakeMatcher(switchStmt, "switchStmt"), this);
}

static StringRef normDeclStmtText(StringRef Text) {
  Text = Text.trim();
  while (Text.consume_back(";")) {
    Text = Text.rtrim();
  }
  return Text;
}

static std::vector<const VarDecl *>
getVarDeclsFromDeclStmt(const DeclStmt *DS) {
  std::vector<const VarDecl *> Vars;
  Vars.reserve(llvm::size(DS->decls()));
  llvm::transform(DS->decls(), std::back_inserter(Vars), [](const auto *D) {
    const auto *VD = dyn_cast<VarDecl>(D);
    assert(VD); // TODO:
    return VD;
  });
  return Vars;
}

void UseInitStatementCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *If = Result.Nodes.getNodeAs<IfStmt>("ifStmt");
  const auto *Switch = Result.Nodes.getNodeAs<SwitchStmt>("switchStmt");
  const auto *PrevDecl = Result.Nodes.getNodeAs<DeclStmt>("prevDecl");
  const auto *Condition = Result.Nodes.getNodeAs<Expr>("condition");
  const auto *Compound = Result.Nodes.getNodeAs<CompoundStmt>("compoundStmt");

  if (!PrevDecl || !Condition || !Compound)
    return;

  const Stmt *Statement = If ? static_cast<const Stmt *>(If) : Switch;
  if (!Statement)
    return;

  if (!allVarsUsedInCondition(PrevDecl, Condition))
    return;

  if (anyVarUsedAfterStmt(PrevDecl, Statement, Compound))
    return;

  const auto AllVarsInDecl = getVarDeclsFromDeclStmt(PrevDecl);
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
              << (AllVarsInDecl.size() == 1) << AllVarsInDecl[0] << !If;
  if (CanFix)
    Diag << FixItHint::CreateRemoval(RemovalRange)
         << FixItHint::CreateInsertion(Condition->getBeginLoc(), NewInitStmt);
}

} // namespace clang::tidy::modernize
