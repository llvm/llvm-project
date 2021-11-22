//=======- UncountedCallArgsChecker.cpp --------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTUtils.h"
#include "DiagOutputUtils.h"
#include "PtrTypesSemantics.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "llvm/ADT/DenseSet.h"

using namespace clang;
using namespace ento;

namespace {

class UncountedCallArgsChecker
    : public Checker<check::ASTDecl<TranslationUnitDecl>> {
  BugType Bug{this,
            "Uncounted call argument for a raw pointer/reference parameter",
            "WebKit coding guidelines"};
  mutable BugReporter *BR;

public:

  void checkASTDecl(const TranslationUnitDecl *TUD, AnalysisManager &MGR,
                    BugReporter &BRArg) const {
    BR = &BRArg;

    // The calls to checkAST* from AnalysisConsumer don't
    // visit template instantiations or lambda classes. We
    // want to visit those, so we make our own RecursiveASTVisitor.
    struct LocalVisitor : public RecursiveASTVisitor<LocalVisitor> {
      const UncountedCallArgsChecker *Checker;
      explicit LocalVisitor(const UncountedCallArgsChecker *Checker)
          : Checker(Checker) {
        assert(Checker);
      }

      bool shouldVisitTemplateInstantiations() const { return true; }
      bool shouldVisitImplicitCode() const { return false; }

      bool VisitCallExpr(const CallExpr *CE) {
        Checker->visitCallExpr(CE);
        return true;
      }
    };

    LocalVisitor visitor(this);
    visitor.TraverseDecl(const_cast<TranslationUnitDecl *>(TUD));
  }

  void visitCallExpr(const CallExpr *CE) const {
    if (shouldSkipCall(CE))
      return;

    if (auto *F = CE->getDirectCallee()) {
      // Skip the first argument for overloaded member operators (e. g. lambda
      // or std::function call operator).
      unsigned ArgIdx = isa<CXXOperatorCallExpr>(CE) && isa_and_nonnull<CXXMethodDecl>(F);

      for (auto P = F->param_begin();
           // FIXME: Also check variadic function parameters.
           // FIXME: Also check default function arguments. Probably a different
           // checker. In case there are default arguments the call can have
           // fewer arguments than the callee has parameters.
           P < F->param_end() && ArgIdx < CE->getNumArgs(); ++P, ++ArgIdx) {
        // TODO: attributes.
        // if ((*P)->hasAttr<SafeRefCntblRawPtrAttr>())
        //  continue;

        const auto *ArgType = (*P)->getType().getTypePtrOrNull();
        if (!ArgType)
          continue; // FIXME? Should we bail?

        // FIXME: more complex types (arrays, references to raw pointers, etc)
        Optional<bool> IsUncounted = isUncountedPtr(ArgType);
        if (!IsUncounted || !(*IsUncounted))
          continue;

        const auto *Arg = CE->getArg(ArgIdx);

        std::pair<const clang::Expr *, bool> ArgOrigin =
            tryToFindPtrOrigin(Arg, true);

        // Temporary ref-counted object created as part of the call argument
        // would outlive the call.
        if (ArgOrigin.second)
          continue;

        if (isa<CXXNullPtrLiteralExpr>(ArgOrigin.first)) {
          // foo(nullptr)
          continue;
        }
        if (isa<IntegerLiteral>(ArgOrigin.first)) {
          // FIXME: Check the value.
          // foo(NULL)
          continue;
        }

        if (isASafeCallArg(ArgOrigin.first))
          continue;

        reportBug(Arg, *P);
      }
    }
  }

  bool shouldSkipCall(const CallExpr *CE) const {
    if (CE->getNumArgs() == 0)
      return false;

    // If an assignment is problematic we should warn about the sole existence
    // of object on LHS.
    if (auto *MemberOp = dyn_cast<CXXOperatorCallExpr>(CE)) {
      // Note: assignemnt to built-in type isn't derived from CallExpr.
      if (MemberOp->isAssignmentOp())
        return false;
    }

    const auto *Callee = CE->getDirectCallee();
    if (!Callee)
      return false;

    auto overloadedOperatorType = Callee->getOverloadedOperator();
    if (overloadedOperatorType == OO_EqualEqual ||
        overloadedOperatorType == OO_ExclaimEqual ||
        overloadedOperatorType == OO_LessEqual ||
        overloadedOperatorType == OO_GreaterEqual ||
        overloadedOperatorType == OO_Spaceship ||
        overloadedOperatorType == OO_AmpAmp ||
        overloadedOperatorType == OO_PipePipe)
      return true;

    if (isCtorOfRefCounted(Callee))
      return true;

    auto name = safeGetName(Callee);
    if (name == "adoptRef" || name == "getPtr" || name == "WeakPtr" ||
        name == "makeWeakPtr" || name == "downcast" || name == "bitwise_cast" ||
        name == "is" || name == "equal" || name == "hash" ||
        name == "isType"
        // FIXME: Most/all of these should be implemented via attributes.
        || name == "equalIgnoringASCIICase" ||
        name == "equalIgnoringASCIICaseCommon" ||
        name == "equalIgnoringNullity")
      return true;

    return false;
  }

  void reportBug(const Expr *CallArg, const ParmVarDecl *Param) const {
    assert(CallArg);

    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);

    const std::string paramName = safeGetName(Param);
    Os << "Call argument";
    if (!paramName.empty()) {
      Os << " for parameter ";
      printQuotedQualifiedName(Os, Param);
    }
    Os << " is uncounted and unsafe.";

    const SourceLocation SrcLocToReport =
        isa<CXXDefaultArgExpr>(CallArg) ? Param->getDefaultArg()->getExprLoc()
                                        : CallArg->getSourceRange().getBegin();

    PathDiagnosticLocation BSLoc(SrcLocToReport, BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    Report->addRange(CallArg->getSourceRange());
    BR->emitReport(std::move(Report));
  }
};
} // namespace

void ento::registerUncountedCallArgsChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<UncountedCallArgsChecker>();
}

bool ento::shouldRegisterUncountedCallArgsChecker(const CheckerManager &) {
  return true;
}
