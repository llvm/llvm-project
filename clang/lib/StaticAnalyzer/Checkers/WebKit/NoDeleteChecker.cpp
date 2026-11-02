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

    const FieldDecl *Field = nullptr;
    const Stmt *OffendingInit = nullptr;
    bool IsCtor = false;
    bool IsDtor = false;
    if (auto *Ctor = dyn_cast<CXXConstructorDecl>(FD)) {
      IsCtor = true;
      Field = TFA.fieldWithNonTrivialCtor(Ctor->getParent());
      if (!Field) {
        for (auto *CtorInit : Ctor->inits()) {
          auto *Init = CtorInit->getInit();
          if (!TFA.isTrivial(Init)) {
            OffendingInit = Init;
            break;
          }
        }
      }
    } else if (auto *Dtor = dyn_cast<CXXDestructorDecl>(FD)) {
      IsDtor = true;
      Field = TFA.fieldWithNonTrivialDtor(Dtor->getParent());
    }

    if (!ParamDecl && !Field && !OffendingInit && TFA.isTrivial(Body))
      return;

    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);

    if (IsCtor)
      Os << "A constructor ";
    else if (IsDtor)
      Os << "A destructor ";
    else
      Os << "A function ";
    printQuotedName(Os, FD);
    // FIXME: Update this to say clang::annotate("webkit.nodelete").
    Os << " has [[clang::annotate_type(\"webkit.nodelete\")]] but it ";
    if (IsCtor && Field)
      Os << "constructs ";
    else if (IsDtor && Field)
      Os << "destructs ";
    else
      Os << "contains ";
    SourceLocation SrcLocToReport;
    SourceRange Range;
    NonTrivialityReason Reason;
    if (ParamDecl) {
      Os << "a parameter ";
      printQuotedName(Os, ParamDecl);
      Os << " which could destruct an object.";
      SrcLocToReport = FD->getBeginLoc();
      Range = ParamDecl->getSourceRange();
    } else if (Field && !OffendingInit) {
      Os << "a member variable ";
      printQuotedName(Os, Field);
      Os << " that could destruct an object.";
      SrcLocToReport = FD->getBeginLoc();
      Range = Field->getSourceRange();
    } else {
      Reason = TrivialFunctionAnalysis::computeReason(
          OffendingInit ? OffendingInit : Body);
      Os << "code that could destruct an object.";
      const Stmt *Offender = Reason.OffendingStmt;
      SrcLocToReport = Offender ? Offender->getBeginLoc() : FD->getBeginLoc();
      Range = Offender ? Offender->getSourceRange() : FD->getSourceRange();
    }

    PathDiagnosticLocation BSLoc(SrcLocToReport, BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    Report->addRange(Range);
    Report->setDeclWithIssue(FD);
    addCallStackNotes(*Report, Reason);
    BR->emitReport(std::move(Report));
  }

  static const FunctionDecl *getDirectCallee(const Stmt *S) {
    if (const auto *CE = dyn_cast_or_null<CallExpr>(S))
      return CE->getDirectCallee();
    if (const auto *CE = dyn_cast_or_null<CXXConstructExpr>(S))
      return CE->getConstructor();
    return nullptr;
  }

  // The offending statement is often just the nearest call to a function that
  // is itself unsafe several levels down. Walk the whole chain of calls, one
  // note per function, down to the code that destructs an object or the
  // function without a visible definition, since that is where the fix belongs.
  void addCallStackNotes(BasicBugReport &Report,
                         const NonTrivialityReason &Reason) const {
    ArrayRef<NonTrivialityReason::Frame> CallStack = Reason.CallStack;
    if (CallStack.empty())
      return;

    // Nothing to add when the offending statement is the call to a function
    // that is opaque or rejected outright; a note would only point back at its
    // declaration, which the primary diagnostic already names.
    const auto &First = CallStack.front();
    const FunctionDecl *Callee = getDirectCallee(Reason.OffendingStmt);
    if (CallStack.size() == 1 && !First.OffendingStmt && Callee &&
        Callee->getCanonicalDecl() == First.Callee->getCanonicalDecl())
      return;

    for (size_t I = 0; I < CallStack.size(); ++I) {
      const FunctionDecl *Fn = CallStack[I].Callee;
      const Stmt *Offender = CallStack[I].OffendingStmt;
      // Implicit special members have nothing worth pointing at.
      if (!Offender && !Fn->getLocation().isValid())
        continue;

      SmallString<100> Buf;
      llvm::raw_svector_ostream Os(Buf);
      // Each note sits on the line it describes, which already spells out the
      // enclosing function and the callee, so name only the callee: naming
      // both would repeat every function in the chain twice.
      if (I + 1 < CallStack.size()) {
        Os << "Calling ";
        printQuotedName(Os, CallStack[I + 1].Callee);
      } else if (Fn->doesThisDeclarationHaveABody()) {
        Os << "Could destruct an object";
      } else {
        printQuotedName(Os, Fn);
        Os << " has no visible definition here, so it is assumed to destruct "
              "an object. Annotate it with "
              "[[clang::annotate_type(\"webkit.nodelete\")]] if it does not.";
      }

      SourceLocation Loc =
          Offender ? Offender->getBeginLoc() : Fn->getLocation();
      SourceRange Range =
          Offender ? Offender->getSourceRange() : Fn->getSourceRange();
      Report.addNote(
          Os.str(), PathDiagnosticLocation(Loc, BR->getSourceManager()), Range);
    }
  }
};

} // namespace

void ento::registerNoDeleteChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<NoDeleteChecker>();
}

bool ento::shouldRegisterNoDeleteChecker(const CheckerManager &) {
  return true;
}
