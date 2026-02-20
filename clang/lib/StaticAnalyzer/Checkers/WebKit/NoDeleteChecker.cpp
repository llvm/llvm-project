//=======- NoDeleteChecker.cpp -----------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DiagOutputUtils.h"
#include "PtrTypesSemantics.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/QualTypeNames.h"
#include "clang/Analysis/DomainSpecific/CocoaConventions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"

using namespace clang;
using namespace ento;

namespace {

class NoDeleteChecker : public Checker<check::ASTDecl<TranslationUnitDecl>> {
  BugType Bug;
  mutable BugReporter *BR = nullptr;
  mutable TrivialFunctionAnalysis TFA;

public:
  NoDeleteChecker()
      : Bug(this,
            "Incorrect [[clang::annotate_type(\"webkit.nodelete\")]] "
            "annotation",
            "WebKit coding guidelines") {}

  void checkASTDecl(const TranslationUnitDecl *TUD, AnalysisManager &MGR,
                    BugReporter &BRArg) const {
    BR = &BRArg;

    // The calls to checkAST* from AnalysisConsumer don't
    // visit template instantiations or lambda classes. We
    // want to visit those, so we make our own visitor.
    struct LocalVisitor final : public ConstDynamicRecursiveASTVisitor {
      const NoDeleteChecker *Checker;
      Decl *DeclWithIssue{nullptr};

      explicit LocalVisitor(const NoDeleteChecker *Checker) : Checker(Checker) {
        assert(Checker);
        ShouldVisitTemplateInstantiations = true;
        ShouldWalkTypesOfTypeLocs = true;
        ShouldVisitImplicitCode = false;
        ShouldVisitLambdaBody = true;
      }

      bool VisitFunctionDecl(const FunctionDecl *FD) override {
        Checker->visitFunctionDecl(FD);
        return true;
      }
    };

    LocalVisitor visitor(this);
    visitor.TraverseDecl(const_cast<TranslationUnitDecl *>(TUD));
  }

  static bool hasNoDeleteAnnotation(const FunctionDecl *FD) {
    if (llvm::any_of(FD->redecls(), isNoDeleteFunction))
      return true;

    const auto *MD = dyn_cast<CXXMethodDecl>(FD);
    if (!MD || !MD->isVirtual())
      return false;

    auto Overriders = llvm::to_vector(MD->overridden_methods());
    while (!Overriders.empty()) {
      const auto *Fn = Overriders.pop_back_val();
      llvm::append_range(Overriders, Fn->overridden_methods());
      if (isNoDeleteFunction(Fn))
        return true;
    }

    return false;
  }

  void visitFunctionDecl(const FunctionDecl *FD) const {
    if (!FD->doesThisDeclarationHaveABody() || FD->isDependentContext())
      return;

    if (!hasNoDeleteAnnotation(FD))
      return;

    auto Body = FD->getBody();
    if (!Body)
      return;

    NamedDecl* ParamDecl = nullptr;
    for (auto* D : FD->parameters()) {
      if (!TFA.hasTrivialDtor(D)) {
        ParamDecl = D;
        break;
      }
    }
    const Stmt *OffendingStmt = nullptr;
    if (!ParamDecl && TFA.isTrivial(Body, &OffendingStmt))
      return;

    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);

    Os << "A function ";
    printQuotedName(Os, FD);
    Os << " has [[clang::annotate_type(\"webkit.nodelete\")]] but it contains ";
    SourceLocation SrcLocToReport;
    SourceRange Range;
    if (ParamDecl) {
      Os << "a parameter ";
      printQuotedName(Os, ParamDecl);
      Os << " which could destruct an object.";
      SrcLocToReport = FD->getBeginLoc();
      Range = ParamDecl->getSourceRange();
    } else {
      Os << "code that could destruct an object.";
      SrcLocToReport = OffendingStmt->getBeginLoc();
      Range = OffendingStmt->getSourceRange();
    }

    PathDiagnosticLocation BSLoc(SrcLocToReport, BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    Report->addRange(Range);
    Report->setDeclWithIssue(FD);
    BR->emitReport(std::move(Report));
  }
};

} // namespace

void ento::registerNoDeleteChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<NoDeleteChecker>();
}

bool ento::shouldRegisterNoDeleteChecker(const CheckerManager &) {
  return true;
}
