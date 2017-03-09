//===--- FunctionSize.cpp - clang-tidy ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FunctionSizeCheck.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

class FunctionASTVisitor : public RecursiveASTVisitor<FunctionASTVisitor> {
  using Base = RecursiveASTVisitor<FunctionASTVisitor>;

public:
  bool TraverseStmt(Stmt *Node) {
    if (!Node)
      return Base::TraverseStmt(Node);

    if (TrackedParent.back() && !isa<CompoundStmt>(Node))
      ++Info.Statements;

    switch (Node->getStmtClass()) {
    case Stmt::IfStmtClass:
    case Stmt::WhileStmtClass:
    case Stmt::DoStmtClass:
    case Stmt::CXXForRangeStmtClass:
    case Stmt::ForStmtClass:
    case Stmt::SwitchStmtClass:
      ++Info.Branches;
    // fallthrough
    case Stmt::CompoundStmtClass:
      TrackedParent.push_back(true);
      break;
    default:
      TrackedParent.push_back(false);
      break;
    }

    Base::TraverseStmt(Node);

    TrackedParent.pop_back();
    return true;
  }

  bool TraverseDecl(Decl *Node) {
    TrackedParent.push_back(false);
    Base::TraverseDecl(Node);
    TrackedParent.pop_back();
    return true;
  }

  struct FunctionInfo {
    FunctionInfo() : Lines(0), Statements(0), Branches(0) {}
    unsigned Lines;
    unsigned Statements;
    unsigned Branches;
  };
  FunctionInfo Info;
  std::vector<bool> TrackedParent;
};

FunctionSizeCheck::FunctionSizeCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      LineThreshold(Options.get("LineThreshold", -1U)),
      StatementThreshold(Options.get("StatementThreshold", 800U)),
      BranchThreshold(Options.get("BranchThreshold", -1U)),
      ParameterThreshold(Options.get("ParameterThreshold", -1U)) {}

void FunctionSizeCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "LineThreshold", LineThreshold);
  Options.store(Opts, "StatementThreshold", StatementThreshold);
  Options.store(Opts, "BranchThreshold", BranchThreshold);
  Options.store(Opts, "ParameterThreshold", ParameterThreshold);
}

void FunctionSizeCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(functionDecl(unless(isInstantiated())).bind("func"), this);
}

void FunctionSizeCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");

  FunctionASTVisitor Visitor;
  Visitor.TraverseDecl(const_cast<FunctionDecl *>(Func));
  auto &FI = Visitor.Info;

  if (FI.Statements == 0)
    return;

  // Count the lines including whitespace and comments. Really simple.
  if (const Stmt *Body = Func->getBody()) {
    SourceManager *SM = Result.SourceManager;
    if (SM->isWrittenInSameFile(Body->getLocStart(), Body->getLocEnd())) {
      FI.Lines = SM->getSpellingLineNumber(Body->getLocEnd()) -
                 SM->getSpellingLineNumber(Body->getLocStart());
    }
  }

  unsigned ActualNumberParameters = Func->getNumParams();

  if (FI.Lines > LineThreshold || FI.Statements > StatementThreshold ||
      FI.Branches > BranchThreshold ||
      ActualNumberParameters > ParameterThreshold) {
    diag(Func->getLocation(),
         "function %0 exceeds recommended size/complexity thresholds")
        << Func;
  }

  if (FI.Lines > LineThreshold) {
    diag(Func->getLocation(),
         "%0 lines including whitespace and comments (threshold %1)",
         DiagnosticIDs::Note)
        << FI.Lines << LineThreshold;
  }

  if (FI.Statements > StatementThreshold) {
    diag(Func->getLocation(), "%0 statements (threshold %1)",
         DiagnosticIDs::Note)
        << FI.Statements << StatementThreshold;
  }

  if (FI.Branches > BranchThreshold) {
    diag(Func->getLocation(), "%0 branches (threshold %1)", DiagnosticIDs::Note)
        << FI.Branches << BranchThreshold;
  }

  if (ActualNumberParameters > ParameterThreshold) {
    diag(Func->getLocation(), "%0 parameters (threshold %1)",
         DiagnosticIDs::Note)
        << ActualNumberParameters << ParameterThreshold;
  }
}

} // namespace readability
} // namespace tidy
} // namespace clang
