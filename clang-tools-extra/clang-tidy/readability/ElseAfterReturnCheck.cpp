//===--- ElseAfterReturnCheck.cpp - clang-tidy-----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ElseAfterReturnCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/FixIt.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

namespace {
static const char ReturnStr[] = "return";
static const char ContinueStr[] = "continue";
static const char BreakStr[] = "break";
static const char ThrowStr[] = "throw";
static const char WarningMessage[] = "do not use 'else' after '%0'";
static const char WarnOnUnfixableStr[] = "WarnOnUnfixable";

const DeclRefExpr *findUsage(const Stmt *Node, int64_t DeclIdentifier) {
  if (!Node)
    return nullptr;
  if (const auto *DeclRef = dyn_cast<DeclRefExpr>(Node)) {
    if (DeclRef->getDecl()->getID() == DeclIdentifier) {
      return DeclRef;
    }
  } else {
    for (const Stmt *ChildNode : Node->children()) {
      if (const DeclRefExpr *Result = findUsage(ChildNode, DeclIdentifier)) {
        return Result;
      }
    }
  }
  return nullptr;
}

const DeclRefExpr *
findUsageRange(const Stmt *Node,
               const llvm::iterator_range<int64_t *> &DeclIdentifiers) {
  if (!Node)
    return nullptr;
  if (const auto *DeclRef = dyn_cast<DeclRefExpr>(Node)) {
    if (llvm::is_contained(DeclIdentifiers, DeclRef->getDecl()->getID())) {
      return DeclRef;
    }
  } else {
    for (const Stmt *ChildNode : Node->children()) {
      if (const DeclRefExpr *Result =
              findUsageRange(ChildNode, DeclIdentifiers)) {
        return Result;
      }
    }
  }
  return nullptr;
}

const DeclRefExpr *checkInitDeclUsageInElse(const IfStmt *If) {
  const auto *InitDeclStmt = dyn_cast_or_null<DeclStmt>(If->getInit());
  if (!InitDeclStmt)
    return nullptr;
  if (InitDeclStmt->isSingleDecl()) {
    const Decl *InitDecl = InitDeclStmt->getSingleDecl();
    assert(isa<VarDecl>(InitDecl) && "SingleDecl must be a VarDecl");
    return findUsage(If->getElse(), InitDecl->getID());
  }
  llvm::SmallVector<int64_t, 4> DeclIdentifiers;
  for (const Decl *ChildDecl : InitDeclStmt->decls()) {
    assert(isa<VarDecl>(ChildDecl) && "Init Decls must be a VarDecl");
    DeclIdentifiers.push_back(ChildDecl->getID());
  }
  return findUsageRange(If->getElse(), DeclIdentifiers);
}

const DeclRefExpr *checkConditionVarUsageInElse(const IfStmt *If) {
  const VarDecl *CondVar = If->getConditionVariable();
  return CondVar != nullptr ? findUsage(If->getElse(), CondVar->getID())
                            : nullptr;
}

bool containsDeclInScope(const Stmt *Node) {
  if (isa<DeclStmt>(Node)) {
    return true;
  }
  if (const auto *Compound = dyn_cast<CompoundStmt>(Node)) {
    return llvm::any_of(Compound->body(), [](const Stmt *SubNode) {
      return isa<DeclStmt>(SubNode);
    });
  }
  return false;
}

void removeElseAndBrackets(DiagnosticBuilder &Diag, ASTContext &Context,
                           const Stmt *Else, SourceLocation ElseLoc) {
  auto Remap = [&](SourceLocation Loc) {
    return Context.getSourceManager().getExpansionLoc(Loc);
  };
  auto TokLen = [&](SourceLocation Loc) {
    return Lexer::MeasureTokenLength(Loc, Context.getSourceManager(),
                                     Context.getLangOpts());
  };

  if (const auto *CS = dyn_cast<CompoundStmt>(Else)) {
    Diag << tooling::fixit::createRemoval(ElseLoc);
    SourceLocation LBrace = CS->getLBracLoc();
    SourceLocation RBrace = CS->getRBracLoc();
    SourceLocation RangeStart =
        Remap(LBrace).getLocWithOffset(TokLen(LBrace) + 1);
    SourceLocation RangeEnd = Remap(RBrace).getLocWithOffset(-1);

    llvm::StringRef Repl = Lexer::getSourceText(
        CharSourceRange::getTokenRange(RangeStart, RangeEnd),
        Context.getSourceManager(), Context.getLangOpts());
    Diag << tooling::fixit::createReplacement(CS->getSourceRange(), Repl);
  } else {
    SourceLocation ElseExpandedLoc = Remap(ElseLoc);
    SourceLocation EndLoc = Remap(Else->getEndLoc());

    llvm::StringRef Repl = Lexer::getSourceText(
        CharSourceRange::getTokenRange(
            ElseExpandedLoc.getLocWithOffset(TokLen(ElseLoc) + 1), EndLoc),
        Context.getSourceManager(), Context.getLangOpts());
    Diag << tooling::fixit::createReplacement(
        SourceRange(ElseExpandedLoc, EndLoc), Repl);
  }
}
} // namespace

ElseAfterReturnCheck::ElseAfterReturnCheck(StringRef Name,
                                           ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      WarnOnUnfixable(Options.get(WarnOnUnfixableStr, true)) {}

void ElseAfterReturnCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, WarnOnUnfixableStr, WarnOnUnfixable);
}

void ElseAfterReturnCheck::registerMatchers(MatchFinder *Finder) {
  const auto InterruptsControlFlow =
      stmt(anyOf(returnStmt().bind(ReturnStr), continueStmt().bind(ContinueStr),
                 breakStmt().bind(BreakStr),
                 expr(ignoringImplicit(cxxThrowExpr().bind(ThrowStr)))));
  Finder->addMatcher(
      compoundStmt(
          forEach(ifStmt(unless(isConstexpr()),
                         hasThen(stmt(
                             anyOf(InterruptsControlFlow,
                                   compoundStmt(has(InterruptsControlFlow))))),
                         hasElse(stmt().bind("else")))
                      .bind("if")))
          .bind("cs"),
      this);
}

void ElseAfterReturnCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *If = Result.Nodes.getNodeAs<IfStmt>("if");
  const auto *Else = Result.Nodes.getNodeAs<Stmt>("else");
  const auto *OuterScope = Result.Nodes.getNodeAs<CompoundStmt>("cs");

  bool IsLastInScope = OuterScope->body_back() == If;
  SourceLocation ElseLoc = If->getElseLoc();

  auto ControlFlowInterruptor = [&]() -> llvm::StringRef {
    for (llvm::StringRef BindingName :
         {ReturnStr, ContinueStr, BreakStr, ThrowStr})
      if (Result.Nodes.getNodeAs<Stmt>(BindingName))
        return BindingName;
    return {};
  }();

  if (!IsLastInScope && containsDeclInScope(Else)) {
    if (WarnOnUnfixable) {
      // Warn, but don't attempt an autofix.
      diag(ElseLoc, WarningMessage) << ControlFlowInterruptor;
    }
    return;
  }

  if (checkConditionVarUsageInElse(If) != nullptr) {
    if (IsLastInScope) {
      // If the if statement is the last statement its enclosing statements
      // scope, we can pull the decl out of the if statement.
      DiagnosticBuilder Diag = diag(ElseLoc, WarningMessage)
                               << ControlFlowInterruptor;
      if (checkInitDeclUsageInElse(If) != nullptr) {
        Diag << tooling::fixit::createReplacement(
                    SourceRange(If->getIfLoc()),
                    (tooling::fixit::getText(*If->getInit(), *Result.Context) +
                     llvm::StringRef("\n"))
                        .str())
             << tooling::fixit::createRemoval(If->getInit()->getSourceRange());
      }
      const DeclStmt *VDeclStmt = If->getConditionVariableDeclStmt();
      const VarDecl *VDecl = If->getConditionVariable();
      std::string Repl =
          (tooling::fixit::getText(*VDeclStmt, *Result.Context) +
           llvm::StringRef(";\n") +
           tooling::fixit::getText(If->getIfLoc(), *Result.Context))
              .str();
      Diag << tooling::fixit::createReplacement(SourceRange(If->getIfLoc()),
                                                Repl)
           << tooling::fixit::createReplacement(VDeclStmt->getSourceRange(),
                                                VDecl->getName());
      removeElseAndBrackets(Diag, *Result.Context, Else, ElseLoc);
    } else if (WarnOnUnfixable) {
      // Warn, but don't attempt an autofix.
      diag(ElseLoc, WarningMessage) << ControlFlowInterruptor;
    }
    return;
  }

  if (checkInitDeclUsageInElse(If) != nullptr) {
    if (IsLastInScope) {
      // If the if statement is the last statement its enclosing statements
      // scope, we can pull the decl out of the if statement.
      DiagnosticBuilder Diag = diag(ElseLoc, WarningMessage)
                               << ControlFlowInterruptor;
      Diag << tooling::fixit::createReplacement(
                  SourceRange(If->getIfLoc()),
                  (tooling::fixit::getText(*If->getInit(), *Result.Context) +
                   "\n" +
                   tooling::fixit::getText(If->getIfLoc(), *Result.Context))
                      .str())
           << tooling::fixit::createRemoval(If->getInit()->getSourceRange());
      removeElseAndBrackets(Diag, *Result.Context, Else, ElseLoc);
    } else if (WarnOnUnfixable) {
      // Warn, but don't attempt an autofix.
      diag(ElseLoc, WarningMessage) << ControlFlowInterruptor;
    }
    return;
  }

  DiagnosticBuilder Diag = diag(ElseLoc, WarningMessage)
                           << ControlFlowInterruptor;
  removeElseAndBrackets(Diag, *Result.Context, Else, ElseLoc);
}

} // namespace readability
} // namespace tidy
} // namespace clang
