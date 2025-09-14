//== BoundsInformationChecker.cpp - bounds information checker --*- C++ -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines BoundsInformationChecker, a path-sensitive checker that
// checks that the buffer and count arguments are within the bounds of
// the source buffer.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/BugReporter/CommonBugCategories.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"

using namespace clang;
using namespace ento;

namespace {
class BoundsInformationChecker : public Checker<check::PreCall> {
  const BugType BT_DifferentMemRegion{
      this, "std::span constructor arguments from different sources",
      categories::SecurityError};
  const BugType BT_NonConstantSizeArg{
      this,
      "std::span constructor for std::array has non-constant size argument",
      categories::SecurityError};
  const BugType BT_OutOfBounds{
      this,
      "std::span constructor for std::array uses out-of-bounds size argument",
      categories::SecurityError};
  void reportBug(ExplodedNode *N, const Expr *E, CheckerContext &C,
                 const BugType &BT, StringRef Msg) const;

public:
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;
};
} // end anonymous namespace

void BoundsInformationChecker::reportBug(ExplodedNode *N, const Expr *E,
                                         CheckerContext &C, const BugType &BT,
                                         StringRef Msg) const {
  // Generate a report for this bug.
  auto R = std::make_unique<PathSensitiveBugReport>(BT, Msg, N);
  if (auto *CE = dyn_cast<CXXConstructExpr>(E)) {
    bugreporter::trackExpressionValue(N, CE->getArg(0), *R);
    bugreporter::trackExpressionValue(N, CE->getArg(1), *R);
  }
  C.emitReport(std::move(R));
}

static const MemRegion *GetRegionOrigin(SVal SV) {
  const SymExpr *Sym = SV.getAsSymbol(/*IncludeBaseRegions =*/true);
  return Sym ? Sym->getOriginRegion() : nullptr;
}

static const ValueDecl *GetExpressionOrigin(const Stmt *STMT) {
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(STMT)) {
    if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl()))
      return VD;
  } else if (const CXXMemberCallExpr *MCE = dyn_cast<CXXMemberCallExpr>(STMT)) {
    if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(
            MCE->getImplicitObjectArgument()->IgnoreParenCasts())) {
      if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl()))
        return VD;
    } else if (const MemberExpr *ME = dyn_cast<MemberExpr>(
                   MCE->getImplicitObjectArgument()->IgnoreParenCasts())) {
      if (const FieldDecl *FD = dyn_cast<FieldDecl>(ME->getMemberDecl()))
        return FD;
    }
  } else if (const CXXOperatorCallExpr *OCE =
                 dyn_cast<CXXOperatorCallExpr>(STMT)) {
    if (OCE->getNumArgs() >= 1) {
      if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(OCE->getArg(0))) {
        if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl()))
          return VD;
      }
    }
  } else if (const UnaryOperator *UnaryOp = dyn_cast<UnaryOperator>(STMT)) {
    if (const ArraySubscriptExpr *ASExpr =
            dyn_cast<ArraySubscriptExpr>(UnaryOp->getSubExpr())) {
      if (const DeclRefExpr *DRE =
              dyn_cast<DeclRefExpr>(ASExpr->getBase()->IgnoreParenCasts())) {
        if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl()))
          return VD;
      }
    }
  } else if (const UnaryExprOrTypeTraitExpr *UTExpr =
                 dyn_cast<UnaryExprOrTypeTraitExpr>(STMT)) {
    if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(
            UTExpr->getArgumentExpr()->IgnoreParenCasts())) {
      if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl()))
        return VD;
    }
  }
  return nullptr;
}

static const ValueDecl *GetConjuredSymbolOrigin(SVal SV) {
  const SymExpr *Sym = SV.getAsSymbol(/*IncludeBaseRegions =*/true);
  if (const SymbolConjured *SCArg = dyn_cast_or_null<SymbolConjured>(Sym)) {
    if (const Stmt *STMTArg = SCArg->getStmt())
      return GetExpressionOrigin(STMTArg);
  }
  return nullptr;
}

void BoundsInformationChecker::checkPreCall(const CallEvent &Call,
                                            CheckerContext &C) const {
  // Return early if not std::span<IT>(IT, size_t) constructor.
  // a. Check if this is a ctor for std::span.
  CallDescription CD({CDM::CXXMethod, {"std", "span", "span"}});
  if (!CD.matches(Call))
    return;
  // b. Check if std::span ctor has two arguments.
  if (Call.getNumArgs() != 2)
    return;
  // c. Check if second std::span ctor argument is of type size_t.
  if (Call.getArgExpr(1)->getType().getCanonicalType() !=
      C.getASTContext().getSizeType())
    return;

  SVal PointerArg = Call.getArgSVal(0);
  SVal SizeArg = Call.getArgSVal(1);

  // If buffer and length params are not from the same "source", then report a
  // bug.
  const MemRegion *MRArg0 = GetRegionOrigin(PointerArg);
  const MemRegion *MRArg1 = GetRegionOrigin(SizeArg);
  if (MRArg0 && MRArg1 && MRArg0->getBaseRegion() != MRArg1->getBaseRegion()) {
    // FIXME: Add more logic to filter out valid cases.
    if (ExplodedNode *N = C.generateNonFatalErrorNode(C.getState())) {
      reportBug(
          N, Call.getOriginExpr(), C, BT_DifferentMemRegion,
          "Constructor args for std::span are from different memory regions");
      return;
    }
  }

  // Check if value comes from an unknown function call.
  const ValueDecl *VDArg0 = GetConjuredSymbolOrigin(PointerArg);
  const ValueDecl *VDArg1 = GetConjuredSymbolOrigin(SizeArg);

  if (VDArg0) {
    // If first argument is std::array.
    // FIXME: Support C arrays.
    if (const auto *CRDecl0 = VDArg0->getType()->getAsCXXRecordDecl()) {
      if (CRDecl0->isInStdNamespace() && CRDecl0->getIdentifier() &&
          CRDecl0->getName() == "array") {
        if (VDArg0 != VDArg1) {
          // Check second argument against known size of std::array.
          if (SizeArg.isConstant()) {
            if (const auto *CTSDecl =
                    dyn_cast<ClassTemplateSpecializationDecl>(CRDecl0)) {
              const TemplateArgumentList &templateArgList =
                  CTSDecl->getTemplateArgs();
              if (templateArgList.size() == 2) {
                const TemplateArgument &templateArg1 = templateArgList[1];
                if (templateArg1.getKind() ==
                        TemplateArgument::ArgKind::Integral &&
                    *SizeArg.getAsInteger() > templateArg1.getAsIntegral()) {
                  if (ExplodedNode *N =
                          C.generateNonFatalErrorNode(C.getState())) {
                    reportBug(N, Call.getOriginExpr(), C, BT_OutOfBounds,
                              "std::span constructed with overflow length");
                    return;
                  }
                }
              }
            }
          } else if (ExplodedNode *N =
                         C.generateNonFatalErrorNode(C.getState())) {
            reportBug(N, Call.getOriginExpr(), C, BT_NonConstantSizeArg,
                      "std::span constructed from std::array with non-constant "
                      "length");
            return;
          }
        }
      }
    }
  }
}

void ento::registerBoundsInformationChecker(CheckerManager &mgr) {
  mgr.registerChecker<BoundsInformationChecker>();
}

bool ento::shouldRegisterBoundsInformationChecker(const CheckerManager &mgr) {
  return true;
}
