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
    if (!FD->doesThisDeclarationHaveABody())
      return;

    bool HasNoDeleteAnnotation = isNoDeleteFunction(FD);
    if (auto *MD = dyn_cast<CXXMethodDecl>(FD)) {
      if (auto *Cls = MD->getParent(); Cls && MD->isVirtual()) {
        CXXBasePaths Paths;
        Paths.setOrigin(Cls);

        Cls->lookupInBases(
            [&](const CXXBaseSpecifier *Base, CXXBasePath &) {
              const Type *T = Base->getType().getTypePtrOrNull();
              if (!T)
                return false;

              const CXXRecordDecl *R = T->getAsCXXRecordDecl();
              for (const CXXMethodDecl *BaseMD : R->methods()) {
                if (BaseMD->getCorrespondingMethodInClass(Cls) == MD) {
                  if (isNoDeleteFunction(FD)) {
                    HasNoDeleteAnnotation = true;
                    return false;
                  }
                }
              }
              return true;
            },
            Paths, /*LookupInDependent =*/true);
      }
    }

    auto Body = FD->getBody();
    if (!Body || TFA.isTrivial(Body))
      return;

    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);

    Os << "A function ";
    printQuotedName(Os, FD);
    Os << " has [[clang::annotate_type(\"webkit.nodelete\")]] but it contains "
          "code that could destruct an object";

    const SourceLocation SrcLocToReport = FD->getBeginLoc();
    PathDiagnosticLocation BSLoc(SrcLocToReport, BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    Report->addRange(FD->getSourceRange());
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
