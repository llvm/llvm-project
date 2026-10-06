//===--- SwitchToIf.cpp - Switch to if refactoring ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactoring/SwitchToIf/SwitchToIf.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Refactoring/AtomicChange.h"
#include "clang/Tooling/Refactoring/RefactoringDiagnostic.h"
#include "clang/Tooling/Refactoring/RefactoringRuleContext.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <optional>

using namespace clang;
using namespace tooling;

namespace {

/// Returns the source text for the given expression.
std::string getExprText(const Expr *E, const SourceManager &SM,
                        const LangOptions &LangOpts) {
  SourceRange Range = E->getSourceRange();
  return Lexer::getSourceText(CharSourceRange::getTokenRange(Range), SM,
                               LangOpts)
      .str();
}

/// Returns the source text for a range.
std::string getSourceText(SourceRange Range, const SourceManager &SM,
                          const LangOptions &LangOpts) {
  return Lexer::getSourceText(CharSourceRange::getTokenRange(Range), SM,
                               LangOpts)
      .str();
}

/// Returns true if the statement is a break statement.
bool isBreakStmt(const Stmt *S) {
  return isa<BreakStmt>(S);
}

/// Recursively collects all statements from a statement, removing breaks.
/// This handles compound statements and stops at the first break.
void collectStatementsWithoutBreaks(const Stmt *S,
                                    SmallVector<const Stmt *, 16> &Result,
                                    const SourceManager &SM,
                                    const LangOptions &LangOpts) {
  if (!S)
    return;
    
  if (isBreakStmt(S))
    return;
    
  if (const CompoundStmt *CS = dyn_cast<CompoundStmt>(S)) {
    // Process each statement in the compound statement
    for (const Stmt *Child : CS->body()) {
      if (isBreakStmt(Child)) {
        // Stop at first break
        break;
      }
      collectStatementsWithoutBreaks(Child, Result, SM, LangOpts);
    }
  } else {
    // For non-compound statements, add them directly
    Result.push_back(S);
  }
}

/// Gets the statements from a case/default, removing breaks.
SmallVector<const Stmt *, 16> getCaseStatements(const SwitchCase *SC,
                                                const SourceManager &SM,
                                                const LangOptions &LangOpts) {
  SmallVector<const Stmt *, 16> Result;
  const Stmt *SubStmt = SC->getSubStmt();
  if (!SubStmt)
    return Result;
  
  collectStatementsWithoutBreaks(SubStmt, Result, SM, LangOpts);
  return Result;
}

} // end anonymous namespace

const RefactoringDescriptor &SwitchToIf::describe() {
  static const RefactoringDescriptor Descriptor = {
      "switch-to-if",
      "Switch to If",
      "Converts a switch statement into an if-else chain",
  };
  return Descriptor;
}

Expected<SwitchToIf>
SwitchToIf::initiate(RefactoringRuleContext &Context,
                     SelectedASTNode Selection) {
  // Find the SwitchStmt in the selection
  const SwitchStmt *Switch = nullptr;
  
  // Helper lambda to recursively search for SwitchStmt
  std::function<const SwitchStmt *(const SelectedASTNode &)> findSwitch =
      [&](const SelectedASTNode &Node) -> const SwitchStmt * {
    if (const SwitchStmt *S = Node.Node.get<SwitchStmt>()) {
      return S;
    }
    // Search in children
    for (const SelectedASTNode &Child : Node.Children) {
      if (const SwitchStmt *S = findSwitch(Child)) {
        return S;
      }
    }
    return nullptr;
  };
  
  Switch = findSwitch(Selection);
  
  if (!Switch) {
    return Context.createDiagnosticError(
        Context.getSelectionRange().getBegin(),
        diag::err_refactor_selection_invalid_ast);
  }
  
  // Validate that the switch has at least one case
  if (!Switch->getSwitchCaseList()) {
    return Context.createDiagnosticError(
        Switch->getSwitchLoc(),
        diag::err_refactor_selection_invalid_ast);
  }
  
  return SwitchToIf(Switch);
}

Expected<AtomicChanges>
SwitchToIf::createSourceReplacements(RefactoringRuleContext &Context) {
  ASTContext &AST = Context.getASTContext();
  SourceManager &SM = AST.getSourceManager();
  const LangOptions &LangOpts = AST.getLangOpts();
  
  const SwitchStmt *Switch = TheSwitch;
  const Expr *Cond = Switch->getCond();
  
  // Get the full source range of the switch statement
  SourceLocation StartLoc = Switch->getBeginLoc();
  SourceLocation EndLoc = Switch->getEndLoc();
  
  // Find the actual end location (closing brace)
  if (const Stmt *Body = Switch->getBody()) {
    EndLoc = Body->getEndLoc();
  }
  
  SourceRange SwitchRange(StartLoc, EndLoc);
  
  // Build the if-else chain
  std::string Replacement;
  llvm::raw_string_ostream OS(Replacement);
  
  std::string CondText = getExprText(Cond, SM, LangOpts);
  
  // Handle init statement if present
  if (Switch->getInit()) {
    std::string InitText = getSourceText(Switch->getInit()->getSourceRange(),
                                         SM, LangOpts);
    OS << InitText << " ";
  }
  
  // Handle condition variable if present
  if (Switch->getConditionVariableDeclStmt()) {
    std::string VarText = getSourceText(
        Switch->getConditionVariableDeclStmt()->getSourceRange(), SM, LangOpts);
    OS << VarText << " ";
  }
  
  bool First = true;
  const SwitchCase *DefaultCase = nullptr;
  SmallVector<const SwitchCase *, 16> Cases;
  
  // Collect all cases and find default
  for (const SwitchCase *SC = Switch->getSwitchCaseList(); SC;
       SC = SC->getNextSwitchCase()) {
    if (isa<DefaultStmt>(SC)) {
      DefaultCase = SC;
    } else {
      Cases.push_back(SC);
    }
  }
  
  // Process cases
  for (const SwitchCase *Case : Cases) {
    if (First) {
      OS << "if (";
      First = false;
    } else {
      OS << " else if (";
    }
    
    const CaseStmt *CS = cast<CaseStmt>(Case);
    const Expr *LHS = CS->getLHS();
    
    // Handle GNU case ranges
    if (CS->caseStmtIsGNURange()) {
      const Expr *RHS = CS->getRHS();
      std::string LHSText = getExprText(LHS, SM, LangOpts);
      std::string RHSText = getExprText(RHS, SM, LangOpts);
      OS << CondText << " >= " << LHSText << " && " << CondText << " <= "
         << RHSText;
    } else {
      std::string CaseValue = getExprText(LHS, SM, LangOpts);
      OS << CondText << " == " << CaseValue;
    }
    
    OS << ") {\n";
    
    // Get statements from this case (without breaks)
    SmallVector<const Stmt *, 16> Statements = getCaseStatements(Case, SM, LangOpts);
    
    // Print statements
    if (Statements.empty()) {
      // Empty case - just add a blank line or comment
      OS << "  // empty case\n";
    } else {
      for (const Stmt *S : Statements) {
        SourceRange StmtRange = S->getSourceRange();
        std::string StmtText = getSourceText(StmtRange, SM, LangOpts);
        
        // Indent the statement
        OS << "  " << StmtText;
        
        // For compound statements, they already have their own braces
        // For other statements, ensure proper termination
        if (!isa<CompoundStmt>(S) && !isa<IfStmt>(S) && !isa<ForStmt>(S) &&
            !isa<WhileStmt>(S) && !isa<SwitchStmt>(S) && !isa<DoStmt>(S) &&
            !isa<BreakStmt>(S) && !isa<ReturnStmt>(S) && !isa<GotoStmt>(S)) {
          // Check if statement already ends with semicolon by looking at the
          // source text
          if (!StmtText.empty() && StmtText.back() != ';') {
            // Try to get the token after the statement
            SourceLocation AfterEnd = Lexer::getLocForEndOfToken(
                StmtRange.getEnd(), 0, SM, LangOpts);
            Token Tok;
            if (Lexer::getRawToken(AfterEnd, Tok, SM, LangOpts, false) ||
                !Tok.is(tok::semi)) {
              OS << ";";
            }
          }
        }
        OS << "\n";
      }
    }
    
    OS << "}";
  }
  
  // Process default case
  if (DefaultCase) {
    if (First) {
      OS << "if (1) { // default case\n";
      First = false;
    } else {
      OS << " else { // default case\n";
    }
    
    SmallVector<const Stmt *, 16> Statements = getCaseStatements(DefaultCase, SM, LangOpts);
    
    if (Statements.empty()) {
      OS << "  // empty default case\n";
    } else {
      for (const Stmt *S : Statements) {
        SourceRange StmtRange = S->getSourceRange();
        std::string StmtText = getSourceText(StmtRange, SM, LangOpts);
        
        OS << "  " << StmtText;
        
        if (!isa<CompoundStmt>(S) && !isa<IfStmt>(S) && !isa<ForStmt>(S) &&
            !isa<WhileStmt>(S) && !isa<SwitchStmt>(S) && !isa<DoStmt>(S) &&
            !isa<BreakStmt>(S) && !isa<ReturnStmt>(S) && !isa<GotoStmt>(S)) {
          if (!StmtText.empty() && StmtText.back() != ';') {
            SourceLocation AfterEnd = Lexer::getLocForEndOfToken(
                StmtRange.getEnd(), 0, SM, LangOpts);
            Token Tok;
            if (Lexer::getRawToken(AfterEnd, Tok, SM, LangOpts, false) ||
                !Tok.is(tok::semi)) {
              OS << ";";
            }
          }
        }
        OS << "\n";
      }
    }
    
    OS << "}";
  }
  
  // Flush the stream to ensure all content is written to Replacement
  OS.flush();
  
  // Create the atomic change
  AtomicChange Change(SM, StartLoc);
  
  // Replace the entire switch statement
  auto Err = Change.replace(SM, CharSourceRange::getTokenRange(SwitchRange),
                            Replacement);
  if (Err)
    return std::move(Err);
  
  return AtomicChanges{std::move(Change)};
}

