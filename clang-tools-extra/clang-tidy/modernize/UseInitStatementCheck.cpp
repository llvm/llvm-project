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
             hasCondition(expr().bind("condition")))
          .bind("ifStmt"),
      this);

  // Matcher for switch statements that use a variable in condition
  Finder->addMatcher(
      switchStmt(unless(isInTemplateInstantiation()),
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
  VariableUsageVisitor CondVisitor(nullptr);
  std::vector<const VarDecl*> VarsInCondition;
  
  // We need to manually traverse to collect variables
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
  
  for (const VarDecl *VD : Collector.Vars) {
    // For each variable used in condition, check if it's declared right before
    const Stmt *Statement = If ? static_cast<const Stmt*>(If) : Switch;
    if (!Statement) continue;
    
    const DeclStmt *PrevDecl = findPreviousDeclStmt(Statement, VD, Result.Context);
    if (!PrevDecl) continue;
    
    // Check that the variable is NOT used in the body (only in condition)
    const Stmt *Body = If ? If->getThen() : Switch->getBody();
    bool usedInBody = Body ? isVariableUsedInStmt(VD, Body) : false;
    
    // Check that variable is NOT used after the statement
    bool usedAfter = isVariableUsedAfterStmt(VD, Statement, Result.Context);
    
    if (!usedInBody && !usedAfter) {
      // Perfect candidate - variable used only in condition
      std::string VarName = VD->getNameAsString();
      std::string InitExpr = Lexer::getSourceText(
          CharSourceRange::getTokenRange(VD->getInit()->getSourceRange()),
          *Result.SourceManager, getLangOpts()).str();
      
      std::string NewInitStmt;
      if (VD->isConstexpr() || VD->getType().isConstQualified()) {
        NewInitStmt = "const auto " + VarName + " = " + InitExpr + "; ";
      } else {
        NewInitStmt = "auto " + VarName + " = " + InitExpr + "; ";
      }
      
      if (If) {
        auto Diag = diag(PrevDecl->getBeginLoc(), 
                        "variable %0 declaration before if statement could be moved into if init statement")
                    << VD
                    << FixItHint::CreateRemoval(PrevDecl->getSourceRange())
                    << FixItHint::CreateInsertion(If->getBeginLoc(), NewInitStmt);
      } else if (Switch) {
        auto Diag = diag(PrevDecl->getBeginLoc(),
                        "variable %0 declaration before switch statement could be moved into switch init statement")  
                    << VD
                    << FixItHint::CreateRemoval(PrevDecl->getSourceRange())
                    << FixItHint::CreateInsertion(Switch->getBeginLoc(), NewInitStmt);
      }
    }
  }
}

} // namespace modernize
} // namespace tidy
} // namespace clang
