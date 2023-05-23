//===--- BranchCloneCheck.cpp - clang-tidy --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BranchCloneCheck.h"
#include "../utils/ASTUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Analysis/CloneDetection.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/Casting.h"

using namespace clang;
using namespace clang::ast_matchers;

namespace {
/// A branch in a switch may consist of several statements; while a branch in
/// an if/else if/else chain is one statement (which may be a CompoundStmt).
using SwitchBranch = llvm::SmallVector<const Stmt *, 2>;
} // anonymous namespace

/// Determines if the bodies of two branches in a switch statements are Type I
/// clones of each other. This function only examines the body of the branch
/// and ignores the `case X:` or `default:` at the start of the branch.
static bool areSwitchBranchesIdentical(const SwitchBranch &LHS,
                                       const SwitchBranch &RHS,
                                       const ASTContext &Context) {
  if (LHS.size() != RHS.size())
    return false;

  for (size_t I = 0, Size = LHS.size(); I < Size; I++) {
    // NOTE: We strip goto labels and annotations in addition to stripping
    // the `case X:` or `default:` labels, but it is very unlikely that this
    // would cause false positives in real-world code.
    if (!tidy::utils::areStatementsIdentical(LHS[I]->stripLabelLikeStatements(),
                                             RHS[I]->stripLabelLikeStatements(),
                                             Context)) {
      return false;
    }
  }

  return true;
}

static bool isFallthroughSwitchBranch(const SwitchBranch &Branch) {
  struct SwitchCaseVisitor : RecursiveASTVisitor<SwitchCaseVisitor> {
    using RecursiveASTVisitor<SwitchCaseVisitor>::DataRecursionQueue;

    bool TraverseLambdaExpr(LambdaExpr *, DataRecursionQueue * = nullptr) {
      return true; // Ignore lambdas
    }

    bool TraverseDecl(Decl *) {
      return true; // No need to check declarations
    }

    bool TraverseSwitchStmt(SwitchStmt *, DataRecursionQueue * = nullptr) {
      return true; // Ignore sub-switches
    }

    bool TraverseSwitchCase(SwitchCase *, DataRecursionQueue * = nullptr) {
      return true; // Ignore cases
    }

    bool TraverseDefaultStmt(DefaultStmt *, DataRecursionQueue * = nullptr) {
      return true; // Ignore defaults
    }

    bool TraverseAttributedStmt(AttributedStmt *S) {
      if (!S)
        return true;

      for (const Attr *A : S->getAttrs()) {
        if (isa<FallThroughAttr>(A))
          return false;
      }

      return true;
    }
  } Visitor;

  for (const Stmt *Elem : Branch) {
    if (!Visitor.TraverseStmt(const_cast<Stmt *>(Elem)))
      return true;
  }
  return false;
}

namespace clang::tidy::bugprone {

void BranchCloneCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      ifStmt(unless(allOf(isConstexpr(), isInTemplateInstantiation())),
             stmt().bind("if"),
             hasParent(stmt(unless(ifStmt(hasElse(equalsBoundNode("if")))))),
             hasElse(stmt().bind("else"))),
      this);
  Finder->addMatcher(switchStmt().bind("switch"), this);
  Finder->addMatcher(conditionalOperator().bind("condOp"), this);
}

void BranchCloneCheck::check(const MatchFinder::MatchResult &Result) {
  const ASTContext &Context = *Result.Context;

  if (const auto *IS = Result.Nodes.getNodeAs<IfStmt>("if")) {
    const Stmt *Then = IS->getThen();
    assert(Then && "An IfStmt must have a `then` branch!");

    const Stmt *Else = Result.Nodes.getNodeAs<Stmt>("else");
    assert(Else && "We only look for `if` statements with an `else` branch!");

    if (!isa<IfStmt>(Else)) {
      // Just a simple if with no `else if` branch.
      if (utils::areStatementsIdentical(Then->IgnoreContainers(),
                                        Else->IgnoreContainers(), Context)) {
        diag(IS->getBeginLoc(), "if with identical then and else branches");
        diag(IS->getElseLoc(), "else branch starts here", DiagnosticIDs::Note);
      }
      return;
    }

    // This is the complicated case when we start an if/else if/else chain.
    // To find all the duplicates, we collect all the branches into a vector.
    llvm::SmallVector<const Stmt *, 4> Branches;
    const IfStmt *Cur = IS;
    while (true) {
      // Store the `then` branch.
      Branches.push_back(Cur->getThen());

      Else = Cur->getElse();
      // The chain ends if there is no `else` branch.
      if (!Else)
        break;

      // Check if there is another `else if`...
      Cur = dyn_cast<IfStmt>(Else);
      if (!Cur) {
        // ...this is just a plain `else` branch at the end of the chain.
        Branches.push_back(Else);
        break;
      }
    }

    size_t N = Branches.size();
    llvm::BitVector KnownAsClone(N);

    for (size_t I = 0; I + 1 < N; I++) {
      // We have already seen Branches[i] as a clone of an earlier branch.
      if (KnownAsClone[I])
        continue;

      int NumCopies = 1;

      for (size_t J = I + 1; J < N; J++) {
        if (KnownAsClone[J] || !utils::areStatementsIdentical(
                                   Branches[I]->IgnoreContainers(),
                                   Branches[J]->IgnoreContainers(), Context))
          continue;

        NumCopies++;
        KnownAsClone[J] = true;

        if (NumCopies == 2) {
          // We report the first occurrence only when we find the second one.
          diag(Branches[I]->getBeginLoc(),
               "repeated branch body in conditional chain");
          SourceLocation End =
              Lexer::getLocForEndOfToken(Branches[I]->getEndLoc(), 0,
                                         *Result.SourceManager, getLangOpts());
          if (End.isValid()) {
            diag(End, "end of the original", DiagnosticIDs::Note);
          }
        }

        diag(Branches[J]->getBeginLoc(), "clone %0 starts here",
             DiagnosticIDs::Note)
            << (NumCopies - 1);
      }
    }
    return;
  }

  if (const auto *CO = Result.Nodes.getNodeAs<ConditionalOperator>("condOp")) {
    // We do not try to detect chains of ?: operators.
    if (utils::areStatementsIdentical(CO->getTrueExpr(), CO->getFalseExpr(),
                                      Context))
      diag(CO->getQuestionLoc(),
           "conditional operator with identical true and false expressions");

    return;
  }

  if (const auto *SS = Result.Nodes.getNodeAs<SwitchStmt>("switch")) {
    const CompoundStmt *Body = dyn_cast_or_null<CompoundStmt>(SS->getBody());

    // Code like
    //   switch (x) case 0: case 1: foobar();
    // is legal and calls foobar() if and only if x is either 0 or 1;
    // but we do not try to distinguish branches in such code.
    if (!Body)
      return;

    // We will first collect the branches of the switch statements. For the
    // sake of simplicity we say that branches are delimited by the SwitchCase
    // (`case:` or `default:`) children of Body; that is, we ignore `case:` or
    // `default:` labels embedded inside other statements and we do not follow
    // the effects of `break` and other manipulation of the control-flow.
    llvm::SmallVector<SwitchBranch, 4> Branches;
    for (const Stmt *S : Body->body()) {
      // If this is a `case` or `default`, we start a new, empty branch.
      if (isa<SwitchCase>(S))
        Branches.emplace_back();

      // There may be code before the first branch (which can be dead code
      // and can be code reached either through goto or through case labels
      // that are embedded inside e.g. inner compound statements); we do not
      // store those statements in branches.
      if (!Branches.empty())
        Branches.back().push_back(S);
    }

    auto *End = Branches.end();
    auto *BeginCurrent = Branches.begin();
    while (BeginCurrent < End) {
      if (isFallthroughSwitchBranch(*BeginCurrent)) {
        ++BeginCurrent;
        continue;
      }

      auto *EndCurrent = BeginCurrent + 1;
      while (EndCurrent < End &&
             areSwitchBranchesIdentical(*BeginCurrent, *EndCurrent, Context)) {
        ++EndCurrent;
      }
      // At this point the iterator range {BeginCurrent, EndCurrent} contains a
      // complete family of consecutive identical branches.

      if (EndCurrent == (BeginCurrent + 1)) {
        // No consecutive identical branches that start on BeginCurrent
        BeginCurrent = EndCurrent;
        continue;
      }

      diag(BeginCurrent->front()->getBeginLoc(),
           "switch has %0 consecutive identical branches")
          << static_cast<int>(std::distance(BeginCurrent, EndCurrent));

      SourceLocation EndLoc = (EndCurrent - 1)->back()->getEndLoc();
      // If the case statement is generated from a macro, it's SourceLocation
      // may be invalid, resulting in an assertion failure down the line.
      // While not optimal, try the begin location in this case, it's still
      // better then nothing.
      if (EndLoc.isInvalid())
        EndLoc = (EndCurrent - 1)->back()->getBeginLoc();
      if (EndLoc.isMacroID())
        EndLoc = Context.getSourceManager().getExpansionLoc(EndLoc);
      EndLoc = Lexer::getLocForEndOfToken(EndLoc, 0, *Result.SourceManager,
                                          getLangOpts());
      if (EndLoc.isValid()) {
        diag(EndLoc, "last of these clones ends here", DiagnosticIDs::Note);
      }
      BeginCurrent = EndCurrent;
    }
    return;
  }

  llvm_unreachable("No if statement and no switch statement.");
}

} // namespace clang::tidy::bugprone
