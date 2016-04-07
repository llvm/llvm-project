//===--- FunctionSize.cpp - clang-tidy ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FunctionSizeCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

FunctionSizeCheck::FunctionSizeCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      LineThreshold(Options.get("LineThreshold", -1U)),
      StatementThreshold(Options.get("StatementThreshold", 800U)),
      BranchThreshold(Options.get("BranchThreshold", -1U)) {}

void FunctionSizeCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "LineThreshold", LineThreshold);
  Options.store(Opts, "StatementThreshold", StatementThreshold);
  Options.store(Opts, "BranchThreshold", BranchThreshold);
}

void FunctionSizeCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      functionDecl(
          unless(isInstantiated()),
          forEachDescendant(
              stmt(unless(compoundStmt()),
                   hasParent(stmt(anyOf(compoundStmt(), ifStmt(),
                                        anyOf(whileStmt(), doStmt(),
                                              cxxForRangeStmt(), forStmt())))))
                  .bind("stmt"))).bind("func"),
      this);
}

void FunctionSizeCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");

  FunctionInfo &FI = FunctionInfos[Func];

  // Count the lines including whitespace and comments. Really simple.
  if (!FI.Lines) {
    if (const Stmt *Body = Func->getBody()) {
      SourceManager *SM = Result.SourceManager;
      if (SM->isWrittenInSameFile(Body->getLocStart(), Body->getLocEnd())) {
        FI.Lines = SM->getSpellingLineNumber(Body->getLocEnd()) -
                   SM->getSpellingLineNumber(Body->getLocStart());
      }
    }
  }

  const auto *Statement = Result.Nodes.getNodeAs<Stmt>("stmt");
  ++FI.Statements;

  // TODO: switch cases, gotos
  if (isa<IfStmt>(Statement) || isa<WhileStmt>(Statement) ||
      isa<ForStmt>(Statement) || isa<SwitchStmt>(Statement) ||
      isa<DoStmt>(Statement) || isa<CXXForRangeStmt>(Statement))
    ++FI.Branches;
}

void FunctionSizeCheck::onEndOfTranslationUnit() {
  // If we're above the limit emit a warning.
  for (const auto &P : FunctionInfos) {
    const FunctionInfo &FI = P.second;
    if (FI.Lines > LineThreshold || FI.Statements > StatementThreshold ||
        FI.Branches > BranchThreshold) {
      diag(P.first->getLocation(),
           "function %0 exceeds recommended size/complexity thresholds")
          << P.first;
    }

    if (FI.Lines > LineThreshold) {
      diag(P.first->getLocation(),
           "%0 lines including whitespace and comments (threshold %1)",
           DiagnosticIDs::Note)
          << FI.Lines << LineThreshold;
    }

    if (FI.Statements > StatementThreshold) {
      diag(P.first->getLocation(), "%0 statements (threshold %1)",
           DiagnosticIDs::Note)
          << FI.Statements << StatementThreshold;
    }

    if (FI.Branches > BranchThreshold) {
      diag(P.first->getLocation(), "%0 branches (threshold %1)",
           DiagnosticIDs::Note)
          << FI.Branches << BranchThreshold;
    }
  }

  FunctionInfos.clear();
}

} // namespace readability
} // namespace tidy
} // namespace clang
