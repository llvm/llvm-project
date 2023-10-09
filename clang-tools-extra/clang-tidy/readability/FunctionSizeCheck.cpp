//===-- FunctionSizeCheck.cpp - clang-tidy --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FunctionSizeCheck.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/BitVector.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {
namespace {

class FunctionASTVisitor : public RecursiveASTVisitor<FunctionASTVisitor> {
  using Base = RecursiveASTVisitor<FunctionASTVisitor>;

public:
  bool VisitVarDecl(VarDecl *VD) {
    // Do not count function params.
    // Do not count decomposition declarations (C++17's structured bindings).
    if (StructNesting == 0 &&
        !(isa<ParmVarDecl>(VD) || isa<DecompositionDecl>(VD)))
      ++Info.Variables;
    return true;
  }
  bool VisitBindingDecl(BindingDecl *BD) {
    // Do count each of the bindings (in the decomposition declaration).
    if (StructNesting == 0)
      ++Info.Variables;
    return true;
  }

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
      [[fallthrough]];
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

  bool TraverseCompoundStmt(CompoundStmt *Node) {
    // If this new compound statement is located in a compound statement, which
    // is already nested NestingThreshold levels deep, record the start location
    // of this new compound statement.
    if (CurrentNestingLevel == Info.NestingThreshold)
      Info.NestingThresholders.push_back(Node->getBeginLoc());

    ++CurrentNestingLevel;
    Base::TraverseCompoundStmt(Node);
    --CurrentNestingLevel;

    return true;
  }

  bool TraverseDecl(Decl *Node) {
    TrackedParent.push_back(false);
    Base::TraverseDecl(Node);
    TrackedParent.pop_back();
    return true;
  }

  bool TraverseLambdaExpr(LambdaExpr *Node) {
    ++StructNesting;
    Base::TraverseLambdaExpr(Node);
    --StructNesting;
    return true;
  }

  bool TraverseCXXRecordDecl(CXXRecordDecl *Node) {
    ++StructNesting;
    Base::TraverseCXXRecordDecl(Node);
    --StructNesting;
    return true;
  }

  bool TraverseStmtExpr(StmtExpr *SE) {
    ++StructNesting;
    Base::TraverseStmtExpr(SE);
    --StructNesting;
    return true;
  }

  struct FunctionInfo {
    unsigned Lines = 0;
    unsigned Statements = 0;
    unsigned Branches = 0;
    unsigned NestingThreshold = 0;
    unsigned Variables = 0;
    std::vector<SourceLocation> NestingThresholders;
  };
  FunctionInfo Info;
  llvm::BitVector TrackedParent;
  unsigned StructNesting = 0;
  unsigned CurrentNestingLevel = 0;
};

} // namespace

FunctionSizeCheck::FunctionSizeCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      LineThreshold(Options.get("LineThreshold", DefaultLineThreshold)),
      StatementThreshold(
          Options.get("StatementThreshold", DefaultStatementThreshold)),
      BranchThreshold(Options.get("BranchThreshold", DefaultBranchThreshold)),
      ParameterThreshold(
          Options.get("ParameterThreshold", DefaultParameterThreshold)),
      NestingThreshold(
          Options.get("NestingThreshold", DefaultNestingThreshold)),
      VariableThreshold(
          Options.get("VariableThreshold", DefaultVariableThreshold)) {}

void FunctionSizeCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "LineThreshold", LineThreshold);
  Options.store(Opts, "StatementThreshold", StatementThreshold);
  Options.store(Opts, "BranchThreshold", BranchThreshold);
  Options.store(Opts, "ParameterThreshold", ParameterThreshold);
  Options.store(Opts, "NestingThreshold", NestingThreshold);
  Options.store(Opts, "VariableThreshold", VariableThreshold);
}

void FunctionSizeCheck::registerMatchers(MatchFinder *Finder) {
  // Lambdas ignored - historically considered part of enclosing function.
  // FIXME: include them instead? Top-level lambdas are currently never counted.
  Finder->addMatcher(functionDecl(unless(isInstantiated()),
                                  unless(cxxMethodDecl(ofClass(isLambda()))))
                         .bind("func"),
                     this);
}

void FunctionSizeCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");

  FunctionASTVisitor Visitor;
  Visitor.Info.NestingThreshold = NestingThreshold.value_or(-1);
  Visitor.TraverseDecl(const_cast<FunctionDecl *>(Func));
  auto &FI = Visitor.Info;

  if (FI.Statements == 0)
    return;

  // Count the lines including whitespace and comments. Really simple.
  if (const Stmt *Body = Func->getBody()) {
    SourceManager *SM = Result.SourceManager;
    if (SM->isWrittenInSameFile(Body->getBeginLoc(), Body->getEndLoc())) {
      FI.Lines = SM->getSpellingLineNumber(Body->getEndLoc()) -
                 SM->getSpellingLineNumber(Body->getBeginLoc());
    }
  }

  unsigned ActualNumberParameters = Func->getNumParams();

  if ((LineThreshold && FI.Lines > LineThreshold) ||
      (StatementThreshold && FI.Statements > StatementThreshold) ||
      (BranchThreshold && FI.Branches > BranchThreshold) ||
      (ParameterThreshold && ActualNumberParameters > ParameterThreshold) ||
      !FI.NestingThresholders.empty() ||
      (VariableThreshold && FI.Variables > VariableThreshold)) {
    diag(Func->getLocation(),
         "function %0 exceeds recommended size/complexity thresholds")
        << Func;
  }

  if (LineThreshold && FI.Lines > LineThreshold) {
    diag(Func->getLocation(),
         "%0 lines including whitespace and comments (threshold %1)",
         DiagnosticIDs::Note)
        << FI.Lines << LineThreshold.value();
  }

  if (StatementThreshold && FI.Statements > StatementThreshold) {
    diag(Func->getLocation(), "%0 statements (threshold %1)",
         DiagnosticIDs::Note)
        << FI.Statements << StatementThreshold.value();
  }

  if (BranchThreshold && FI.Branches > BranchThreshold) {
    diag(Func->getLocation(), "%0 branches (threshold %1)", DiagnosticIDs::Note)
        << FI.Branches << BranchThreshold.value();
  }

  if (ParameterThreshold && ActualNumberParameters > ParameterThreshold) {
    diag(Func->getLocation(), "%0 parameters (threshold %1)",
         DiagnosticIDs::Note)
        << ActualNumberParameters << ParameterThreshold.value();
  }

  for (const auto &CSPos : FI.NestingThresholders) {
    diag(CSPos, "nesting level %0 starts here (threshold %1)",
         DiagnosticIDs::Note)
        << NestingThreshold.value() + 1 << NestingThreshold.value();
  }

  if (VariableThreshold && FI.Variables > VariableThreshold) {
    diag(Func->getLocation(), "%0 variables (threshold %1)",
         DiagnosticIDs::Note)
        << FI.Variables << VariableThreshold.value();
  }
}

} // namespace clang::tidy::readability
