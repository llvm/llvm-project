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
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/FixIt.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

namespace {

class PPConditionalCollector : public PPCallbacks {
public:
  PPConditionalCollector(
      ElseAfterReturnCheck::ConditionalBranchMap &Collections,
      const SourceManager &SM)
      : Collections(Collections), SM(SM) {}
  void Endif(SourceLocation Loc, SourceLocation IfLoc) override {
    if (!SM.isWrittenInSameFile(Loc, IfLoc))
      return;
    SmallVectorImpl<SourceRange> &Collection = Collections[SM.getFileID(Loc)];
    assert(Collection.empty() || Collection.back().getEnd() < Loc);
    Collection.emplace_back(IfLoc, Loc);
  }

private:
  ElseAfterReturnCheck::ConditionalBranchMap &Collections;
  const SourceManager &SM;
};

} // namespace

static const char InterruptingStr[] = "interrupting";
static const char WarningMessage[] = "do not use 'else' after '%0'";
static const char WarnOnUnfixableStr[] = "WarnOnUnfixable";
static const char WarnOnConditionVariablesStr[] = "WarnOnConditionVariables";

static const DeclRefExpr *findUsage(const Stmt *Node, int64_t DeclIdentifier) {
  if (!Node)
    return nullptr;
  if (const auto *DeclRef = dyn_cast<DeclRefExpr>(Node)) {
    if (DeclRef->getDecl()->getID() == DeclIdentifier)
      return DeclRef;
  } else {
    for (const Stmt *ChildNode : Node->children()) {
      if (const DeclRefExpr *Result = findUsage(ChildNode, DeclIdentifier))
        return Result;
    }
  }
  return nullptr;
}

static const DeclRefExpr *
findUsageRange(const Stmt *Node,
               const llvm::ArrayRef<int64_t> &DeclIdentifiers) {
  if (!Node)
    return nullptr;
  if (const auto *DeclRef = dyn_cast<DeclRefExpr>(Node)) {
    if (llvm::is_contained(DeclIdentifiers, DeclRef->getDecl()->getID()))
      return DeclRef;
  } else {
    for (const Stmt *ChildNode : Node->children()) {
      if (const DeclRefExpr *Result =
              findUsageRange(ChildNode, DeclIdentifiers))
        return Result;
    }
  }
  return nullptr;
}

static const DeclRefExpr *checkInitDeclUsageInElse(const IfStmt *If) {
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

static const DeclRefExpr *checkConditionVarUsageInElse(const IfStmt *If) {
  if (const VarDecl *CondVar = If->getConditionVariable())
    return findUsage(If->getElse(), CondVar->getID());
  return nullptr;
}

static bool containsDeclInScope(const Stmt *Node) {
  if (isa<DeclStmt>(Node))
    return true;
  if (const auto *Compound = dyn_cast<CompoundStmt>(Node))
    return llvm::any_of(Compound->body(), [](const Stmt *SubNode) {
      return isa<DeclStmt>(SubNode);
    });
  return false;
}

static void removeElseAndBrackets(DiagnosticBuilder &Diag, ASTContext &Context,
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

ElseAfterReturnCheck::ElseAfterReturnCheck(StringRef Name,
                                           ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      WarnOnUnfixable(Options.get(WarnOnUnfixableStr, true)),
      WarnOnConditionVariables(Options.get(WarnOnConditionVariablesStr, true)) {
}

void ElseAfterReturnCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, WarnOnUnfixableStr, WarnOnUnfixable);
  Options.store(Opts, WarnOnConditionVariablesStr, WarnOnConditionVariables);
}

void ElseAfterReturnCheck::registerPPCallbacks(const SourceManager &SM,
                                               Preprocessor *PP,
                                               Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(
      std::make_unique<PPConditionalCollector>(this->PPConditionals, SM));
}

void ElseAfterReturnCheck::registerMatchers(MatchFinder *Finder) {
  const auto InterruptsControlFlow = stmt(anyOf(
      returnStmt().bind(InterruptingStr), continueStmt().bind(InterruptingStr),
      breakStmt().bind(InterruptingStr), cxxThrowExpr().bind(InterruptingStr)));
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

static bool hasPreprocessorBranchEndBetweenLocations(
    const ElseAfterReturnCheck::ConditionalBranchMap &ConditionalBranchMap,
    const SourceManager &SM, SourceLocation StartLoc, SourceLocation EndLoc) {

  SourceLocation ExpandedStartLoc = SM.getExpansionLoc(StartLoc);
  SourceLocation ExpandedEndLoc = SM.getExpansionLoc(EndLoc);
  if (!SM.isWrittenInSameFile(ExpandedStartLoc, ExpandedEndLoc))
    return false;

  // StartLoc and EndLoc expand to the same macro.
  if (ExpandedStartLoc == ExpandedEndLoc)
    return false;

  assert(ExpandedStartLoc < ExpandedEndLoc);

  auto Iter = ConditionalBranchMap.find(SM.getFileID(ExpandedEndLoc));

  if (Iter == ConditionalBranchMap.end() || Iter->getSecond().empty())
    return false;

  const SmallVectorImpl<SourceRange> &ConditionalBranches = Iter->getSecond();

  assert(llvm::is_sorted(ConditionalBranches,
                         [](const SourceRange &LHS, const SourceRange &RHS) {
                           return LHS.getEnd() < RHS.getEnd();
                         }));

  // First conditional block that ends after ExpandedStartLoc.
  const auto *Begin =
      llvm::lower_bound(ConditionalBranches, ExpandedStartLoc,
                        [](const SourceRange &LHS, const SourceLocation &RHS) {
                          return LHS.getEnd() < RHS;
                        });
  const auto *End = ConditionalBranches.end();
  for (; Begin != End && Begin->getEnd() < ExpandedEndLoc; ++Begin)
    if (Begin->getBegin() < ExpandedStartLoc)
      return true;
  return false;
}

static StringRef getControlFlowString(const Stmt &Stmt) {
  if (isa<ReturnStmt>(Stmt))
    return "return";
  if (isa<ContinueStmt>(Stmt))
    return "continue";
  if (isa<BreakStmt>(Stmt))
    return "break";
  if (isa<CXXThrowExpr>(Stmt))
    return "throw";
  llvm_unreachable("Unknown control flow interruptor");
}

void ElseAfterReturnCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *If = Result.Nodes.getNodeAs<IfStmt>("if");
  const auto *Else = Result.Nodes.getNodeAs<Stmt>("else");
  const auto *OuterScope = Result.Nodes.getNodeAs<CompoundStmt>("cs");
  const auto *Interrupt = Result.Nodes.getNodeAs<Stmt>(InterruptingStr);
  SourceLocation ElseLoc = If->getElseLoc();

  if (hasPreprocessorBranchEndBetweenLocations(
          PPConditionals, *Result.SourceManager, Interrupt->getBeginLoc(),
          ElseLoc))
    return;

  bool IsLastInScope = OuterScope->body_back() == If;
  StringRef ControlFlowInterruptor = getControlFlowString(*Interrupt);

  if (!IsLastInScope && containsDeclInScope(Else)) {
    if (WarnOnUnfixable) {
      // Warn, but don't attempt an autofix.
      diag(ElseLoc, WarningMessage) << ControlFlowInterruptor;
    }
    return;
  }

  if (checkConditionVarUsageInElse(If) != nullptr) {
    if (!WarnOnConditionVariables)
      return;
    if (IsLastInScope) {
      // If the if statement is the last statement of its enclosing statements
      // scope, we can pull the decl out of the if statement.
      DiagnosticBuilder Diag = diag(ElseLoc, WarningMessage)
                               << ControlFlowInterruptor
                               << SourceRange(ElseLoc);
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
    if (!WarnOnConditionVariables)
      return;
    if (IsLastInScope) {
      // If the if statement is the last statement of its enclosing statements
      // scope, we can pull the decl out of the if statement.
      DiagnosticBuilder Diag = diag(ElseLoc, WarningMessage)
                               << ControlFlowInterruptor
                               << SourceRange(ElseLoc);
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
                           << ControlFlowInterruptor << SourceRange(ElseLoc);
  removeElseAndBrackets(Diag, *Result.Context, Else, ElseLoc);
}

} // namespace readability
} // namespace tidy
} // namespace clang
