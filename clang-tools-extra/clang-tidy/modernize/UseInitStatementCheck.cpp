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
#include <cctype>
#include <map>
#include <optional>

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

} // namespace

void UseInitStatementCheck::registerMatchers(MatchFinder *Finder) {
  // Matcher for if statements that use a variable in condition
  Finder->addMatcher(ifStmt(unless(isInTemplateInstantiation()),
                            unless(hasInitStatement(anything())),
                            hasCondition(expr().bind("condition")))
                         .bind("ifStmt"),
                     this);

  // Matcher for switch statements that use a variable in condition
  Finder->addMatcher(switchStmt(unless(isInTemplateInstantiation()),
                                unless(hasInitStatement(anything())),
                                hasCondition(expr().bind("condition")))
                         .bind("switchStmt"),
                     this);
}

static const DeclStmt *findPreviousDeclStmt(const Stmt *CurrentStmt,
                                            ASTContext *Context) {
  // Validate inputs
  if (!CurrentStmt) {
    return nullptr;
  }

  // Get parent compound statement
  const auto &Parents = Context->getParents(*CurrentStmt);
  if (Parents.empty()) {
    return nullptr;
  }

  const auto *ParentCompound = Parents[0].get<CompoundStmt>();
  if (!ParentCompound) {
    return nullptr;
  }

  // Find current statement position
  const auto Statements = ParentCompound->body();
  const auto CurrentPosition = llvm::find(Statements, CurrentStmt);

  if (CurrentPosition == Statements.end() ||
      CurrentPosition == Statements.begin()) {
    return nullptr; // Not found or no previous statement
  }

  // Get previous statement
  const auto *PreviousStatement = *std::prev(CurrentPosition);
  return dyn_cast<DeclStmt>(PreviousStatement);
}

static bool isVariableUsedAfterStmt(const VarDecl *VD, const Stmt *STMT,
                                    ASTContext *Context) {
  // Validate inputs
  if (!STMT || !VD) {
    return false;
  }

  // Get parent compound statement
  const auto &Parents = Context->getParents(*STMT);
  if (Parents.empty()) {
    return false;
  }

  const auto *ParentCompound = Parents[0].get<CompoundStmt>();
  if (!ParentCompound) {
    return false;
  }

  // Find current statement in the compound statement
  const auto &Statements = ParentCompound->body();
  const auto CurrentPosition = llvm::find(Statements, STMT);

  // Check if statement was found and there are statements after it
  if (CurrentPosition == Statements.end()) {
    return false;
  }

  const auto StatementsAfter = std::next(CurrentPosition);
  if (StatementsAfter == Statements.end()) {
    return false; // No statements after current one
  }

  // Check each subsequent statement for variable usage
  return std::any_of(StatementsAfter, Statements.end(), [&](const Stmt *S) {
    VariableUsageVisitor Visitor(VD);
    Visitor.TraverseStmt(const_cast<Stmt *>(S));
    return Visitor.foundUsage();
  });
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
  const auto *Condition = Result.Nodes.getNodeAs<Expr>("condition");

  const Stmt *Statement = If ? static_cast<const Stmt *>(If) : Switch;
  if (!Statement)
    return;

  // Find all variable references in the condition
  class VarCollector : public RecursiveASTVisitor<VarCollector> {
  public:
    std::vector<const VarDecl *> Vars;

    bool VisitDeclRefExpr(DeclRefExpr *DRE) {
      if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
        Vars.push_back(VD);
      }
      return true; // Continue traversal
    }
  };

  VarCollector Collector;
  Collector.TraverseStmt(const_cast<Expr *>(Condition));

  const DeclStmt *PrevDecl = findPreviousDeclStmt(Statement, Result.Context);

  // Get all variables declared in this DeclStmt
  std::vector<const VarDecl *> AllVarsInDecl;
  AllVarsInDecl.reserve(llvm::size(PrevDecl->decls()));
  for (const auto *D : PrevDecl->decls()) {
    if (const auto *VD = dyn_cast<VarDecl>(D))
      AllVarsInDecl.push_back(VD);
  }

  // TODO: do we need to check all variables??
  // Check if all variables in DeclStmt are used in condition
  const auto varUnusedInCondition =
      llvm::find_if_not(AllVarsInDecl, [&](const VarDecl *VD) {
        return llvm::is_contained(Collector.Vars, VD);
      });

  if (varUnusedInCondition != AllVarsInDecl.end())
    return;

  // Check that none of the variables are used after the statement
  const auto anyUsageAfter =
      llvm::find_if(AllVarsInDecl, [&](const VarDecl *VD) {
        return isVariableUsedAfterStmt(VD, Statement, Result.Context);
      });

  if (anyUsageAfter != AllVarsInDecl.end())
    return;

  // All conditions met - suggest moving the entire DeclStmt
  // Get the source range including the semicolon
  const SourceRange RemovalRange = PrevDecl->getSourceRange();

  // Check if the range can be fixed (i.e., doesn't contain macro expansions)
  const bool CanFix =
      utils::rangeCanBeFixed(RemovalRange, Result.SourceManager) &&
      !Condition->getBeginLoc().isMacroID();

  // Get the text of the declaration (without semicolon)
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
