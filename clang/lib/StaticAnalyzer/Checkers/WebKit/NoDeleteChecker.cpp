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

  void visitFunctionDecl(const FunctionDecl *FD) const {
    if (!FD->doesThisDeclarationHaveABody() || FD->isDependentContext())
      return;

    if (!isNoDeleteFunction(FD))
      return;

    auto Body = FD->getBody();
    if (!Body)
      return;

    NamedDecl *ParamDecl = nullptr;
    for (auto *D : FD->parameters()) {
      if (!TFA.hasTrivialDtor(D)) {
        ParamDecl = D;
        break;
      }
    }
    if (!ParamDecl && TFA.isTrivial(Body))
      return;

    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);

    Os << "A function ";
    printQuotedName(Os, FD);
    Os << " has [[clang::annotate_type(\"webkit.nodelete\")]] but it contains ";
    SourceLocation SrcLocToReport;
    SourceRange Range;
    NonTrivialityReason Reason;
    if (ParamDecl) {
      Os << "a parameter ";
      printQuotedName(Os, ParamDecl);
      Os << " which could destruct an object.";
      SrcLocToReport = FD->getBeginLoc();
      Range = ParamDecl->getSourceRange();
    } else {
      Reason = TrivialFunctionAnalysis::explainNonTriviality(Body);
      Os << "code that could destruct an object.";
      // The analysis found no statement it could point at, which leaves the
      // whole function as the only honest location.
      const Stmt *Offender = Reason.OffendingStmt;
      SrcLocToReport = Offender ? Offender->getBeginLoc() : FD->getBeginLoc();
      Range = Offender ? Offender->getSourceRange() : FD->getSourceRange();
    }

    PathDiagnosticLocation BSLoc(SrcLocToReport, BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    Report->addRange(Range);
    Report->setDeclWithIssue(FD);
    addRootCauseNote(*Report, Reason);
    BR->emitReport(std::move(Report));
  }

  // The function a call expression invokes directly, if any.
  static const FunctionDecl *getDirectCallee(const Stmt *S) {
    if (const auto *CE = dyn_cast_or_null<CallExpr>(S))
      return CE->getDirectCallee();
    if (const auto *CE = dyn_cast_or_null<CXXConstructExpr>(S))
      return CE->getConstructor();
    return nullptr;
  }

  // The offending statement is often just the nearest call to a function that
  // is itself unsafe several levels down. Point at the function at the bottom
  // of that chain, since that is where the fix belongs.
  void addRootCauseNote(BasicBugReport &Report,
                        const NonTrivialityReason &Reason) const {
    const FunctionDecl *RootCause = Reason.RootCause;
    // Implicit special members have nothing worth pointing at.
    if (!RootCause || !RootCause->getLocation().isValid())
      return;

    // Nothing to add when the offending statement is the call to the root
    // cause; the primary diagnostic already points right at it.
    const FunctionDecl *Callee = getDirectCallee(Reason.OffendingStmt);
    if (Callee && Callee->getCanonicalDecl() == RootCause->getCanonicalDecl())
      return;

    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);
    printQuotedName(Os, RootCause);
    if (RootCause->doesThisDeclarationHaveABody()) {
      Os << " could destruct an object.";
    } else {
      Os << " has no visible definition here, so it is assumed to destruct an "
            "object. Annotate it with "
            "[[clang::annotate_type(\"webkit.nodelete\")]] if it does not.";
    }

    PathDiagnosticLocation Loc(RootCause->getLocation(),
                               BR->getSourceManager());
    Report.addNote(Os.str(), Loc, RootCause->getSourceRange());
  }
};

} // namespace

void ento::registerNoDeleteChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<NoDeleteChecker>();
}

bool ento::shouldRegisterNoDeleteChecker(const CheckerManager &) {
  return true;
}
