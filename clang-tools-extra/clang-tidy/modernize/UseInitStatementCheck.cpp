//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseInitStatementCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "../utils/LexerUtils.h"
#include "../utils/ASTUtils.h"
#include <algorithm>
#include <cctype>
#include <map>

using namespace clang;
using namespace ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

namespace {

class VariableUsageVisitor : public RecursiveASTVisitor<VariableUsageVisitor> {
public:
  explicit VariableUsageVisitor(const VarDecl *TargetVar) : TargetVar(TargetVar) {}
  
  bool VisitDeclRefExpr(DeclRefExpr *DRE) {
    if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      if (VD == TargetVar) {
        FoundUsage = true;
        return false; // Stop traversal
      }
    }
    return true;
  }
  
  bool foundUsage() const { return FoundUsage; }
  
  // Reset for reuse
  void reset() { FoundUsage = false; }

private:
  const VarDecl *TargetVar;
  bool FoundUsage = false;
};

} // namespace

void UseInitStatementCheck::registerMatchers(MatchFinder *Finder) {
  // Matcher for if statements that use a variable in condition
  Finder->addMatcher(
      ifStmt(unless(isInTemplateInstantiation()),
             unless(hasInitStatement(anything())),
             hasCondition(expr().bind("condition")))
          .bind("ifStmt"),
      this);

  // Matcher for switch statements that use a variable in condition
  Finder->addMatcher(
      switchStmt(unless(isInTemplateInstantiation()),
                 unless(hasInitStatement(anything())),
                 hasCondition(expr().bind("condition")))
          .bind("switchStmt"),
      this);
}

const DeclStmt* UseInitStatementCheck::findPreviousDeclStmt(const Stmt *CurrentStmt, 
                                                           const VarDecl *TargetVar,
                                                           ASTContext *Context) {
  if (!CurrentStmt || !TargetVar) return nullptr;
  
  // Get the parent compound statement
  const auto &Parent = Context->getParents(*CurrentStmt);
  if (Parent.empty()) return nullptr;
  
  if (const auto *CS = Parent[0].get<CompoundStmt>()) {
    const Stmt *PrevStmt = nullptr;
    
    // Find the current statement in the compound statement
    for (const auto *S : CS->body()) {
      if (S == CurrentStmt) {
        // Found current statement, check previous one
        if (PrevStmt) {
          if (const auto *DS = dyn_cast<DeclStmt>(PrevStmt)) {
            // Check if this DeclStmt declares our target variable
            for (const auto *D : DS->decls()) {
              if (const auto *VD = dyn_cast<VarDecl>(D)) {
                if (VD == TargetVar) {
                  return DS;
                }
              }
            }
          }
        }
        break;
      }
      PrevStmt = S;
    }
  }
  
  return nullptr;
}

bool UseInitStatementCheck::isVariableUsedInStmt(const VarDecl *VD, const Stmt *S) {
  if (!S || !VD) return false;
  
  VariableUsageVisitor Visitor(VD);
  Visitor.TraverseStmt(const_cast<Stmt*>(S));
  return Visitor.foundUsage();
}

bool UseInitStatementCheck::isVariableUsedAfterStmt(const VarDecl *VD, const Stmt *Stmt, 
                                                   ASTContext *Context) {
  if (!Stmt || !VD) return false;
  
  // Get the parent compound statement
  const auto &Parent = Context->getParents(*Stmt);
  if (Parent.empty()) return false;
  
  if (const auto *CS = Parent[0].get<CompoundStmt>()) {
    bool foundCurrent = false;
    
    // Look for variable usage in statements after the current one
    for (const auto *S : CS->body()) {
      if (S == Stmt) {
        foundCurrent = true;
        continue;
      }
      
      if (foundCurrent) {
        // Check if variable is used in this statement
        if (isVariableUsedInStmt(VD, S)) {
          return true;
        }
      }
    }
  }
  
  return false;
}

void UseInitStatementCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *If = Result.Nodes.getNodeAs<IfStmt>("ifStmt");
  const auto *Switch = Result.Nodes.getNodeAs<SwitchStmt>("switchStmt");
  const auto *Condition = Result.Nodes.getNodeAs<Expr>("condition");
  
  if (!Condition) return;
  
  // Find all variable references in the condition
  class VarCollector : public RecursiveASTVisitor<VarCollector> {
  public:
    std::vector<const VarDecl*> Vars;
    
    bool VisitDeclRefExpr(DeclRefExpr *DRE) {
      if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
        Vars.push_back(VD);
      }
      return true;
    }
  };
  
  VarCollector Collector;
  Collector.TraverseStmt(const_cast<Expr*>(Condition));
  
  const Stmt *Statement = If ? static_cast<const Stmt*>(If) : Switch;
  if (!Statement) return;
  
  // Group variables by their DeclStmt
  std::map<const DeclStmt*, std::vector<const VarDecl*>> DeclStmtToVars;
  
  for (const VarDecl *VD : Collector.Vars) {
    const DeclStmt *PrevDecl = findPreviousDeclStmt(Statement, VD, Result.Context);
    if (PrevDecl) {
      DeclStmtToVars[PrevDecl].push_back(VD);
    }
  }
  
  // Process each DeclStmt
  for (const auto &Pair : DeclStmtToVars) {
    const DeclStmt *PrevDecl = Pair.first;
    const std::vector<const VarDecl*> &VarsInDecl = Pair.second;
    
    // Get all variables declared in this DeclStmt
    std::vector<const VarDecl*> AllVarsInDecl;
    for (const auto *D : PrevDecl->decls()) {
      if (const auto *VD = dyn_cast<VarDecl>(D)) {
        AllVarsInDecl.push_back(VD);
      }
    }
    
    // Check if all variables in DeclStmt are used in condition
    bool allVarsUsedInCondition = true;
    for (const VarDecl *VD : AllVarsInDecl) {
      if (std::find(VarsInDecl.begin(), VarsInDecl.end(), VD) == VarsInDecl.end()) {
        allVarsUsedInCondition = false;
        break;
      }
    }
    
    if (!allVarsUsedInCondition) continue;
    
    // Check that none of the variables are used after the statement
    bool anyUsedAfter = false;
    for (const VarDecl *VD : AllVarsInDecl) {
      if (isVariableUsedAfterStmt(VD, Statement, Result.Context)) {
        anyUsedAfter = true;
        break;
      }
    }
    
    if (anyUsedAfter) continue;
    
    // All conditions met - suggest moving the entire DeclStmt
    // Get the source range including the semicolon
    SourceRange RemovalRange = PrevDecl->getSourceRange();
    std::optional<Token> Semicolon = utils::lexer::findNextTokenSkippingComments(
        PrevDecl->getEndLoc(), *Result.SourceManager, getLangOpts());
    if (Semicolon && Semicolon->is(tok::semi)) {
      RemovalRange.setEnd(Semicolon->getEndLoc());
    }
    
    // Check if the range can be fixed (i.e., doesn't contain macro expansions)
    bool CanFix = utils::rangeCanBeFixed(RemovalRange, Result.SourceManager);
    
    // Also check if the insertion location is in a macro
    if (CanFix && Condition->getBeginLoc().isMacroID()) {
      CanFix = false;
    }
    
    // Get the text of the declaration (without semicolon)
    std::string DeclStmtText = Lexer::getSourceText(
        CharSourceRange::getTokenRange(PrevDecl->getSourceRange()),
        *Result.SourceManager, getLangOpts()).str();
    
    // Remove trailing semicolon and whitespace if present
    while (!DeclStmtText.empty() && 
           (DeclStmtText.back() == ';' || std::isspace(DeclStmtText.back()))) {
      DeclStmtText.pop_back();
    }
    
    const std::string NewInitStmt = DeclStmtText + "; ";

    if (If) {
      std::string Message = AllVarsInDecl.size() > 1 
          ? "multiple variable declaration before if statement could be moved into if init statement"
          : "variable %0 declaration before if statement could be moved into if init statement";
      auto Diag = diag(PrevDecl->getBeginLoc(), Message);
      if (AllVarsInDecl.size() == 1) {
        Diag << AllVarsInDecl[0];
      }
      if (CanFix) {
        Diag << FixItHint::CreateRemoval(RemovalRange)
             << FixItHint::CreateInsertion(Condition->getBeginLoc(), NewInitStmt);
      }
    } else if (Switch) {
      std::string Message = AllVarsInDecl.size() > 1
          ? "multiple variable declaration before switch statement could be moved into switch init statement"
          : "variable %0 declaration before switch statement could be moved into switch init statement";
      auto Diag = diag(PrevDecl->getBeginLoc(), Message);
      if (AllVarsInDecl.size() == 1) {
        Diag << AllVarsInDecl[0];
      }
      if (CanFix) {
        Diag << FixItHint::CreateRemoval(RemovalRange)
             << FixItHint::CreateInsertion(Condition->getBeginLoc(), NewInitStmt);
      }
    }
  }
}

} // namespace modernize
} // namespace tidy
} // namespace clang
