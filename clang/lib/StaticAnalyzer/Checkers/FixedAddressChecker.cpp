//=== FixedAddressChecker.cpp - Fixed address usage checker ----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files defines FixedAddressChecker, a builtin checker that checks for
// assignment of a fixed address to a pointer.
// Using a fixed address is not portable because that address will probably
// not be valid in all environments or platforms.
// This check corresponds to CWE-587.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {
class FixedAddressChecker
    : public Checker<check::PreStmt<BinaryOperator>, check::PreStmt<DeclStmt>,
                     check::PreCall> {
  const BugType BT{this, "Use fixed address"};

  void checkUseOfFixedAddress(QualType DstType, const Expr *SrcExpr,
                              CheckerContext &C) const;

public:
  void checkPreStmt(const BinaryOperator *BO, CheckerContext &C) const;
  void checkPreStmt(const DeclStmt *DS, CheckerContext &C) const;
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;
};
}

void FixedAddressChecker::checkUseOfFixedAddress(QualType DstType,
                                                 const Expr *SrcExpr,
                                                 CheckerContext &C) const {
  if (!DstType->isPointerType())
    return;

  if (SrcExpr->IgnoreParenCasts()->getType()->isPointerType())
    return;

  SVal RV = C.getSVal(SrcExpr);

  if (!RV.isConstant() || RV.isZeroConstant())
    return;

  if (C.getSourceManager().isInSystemMacro(SrcExpr->getBeginLoc()))
    return;

  if (ExplodedNode *N = C.generateNonFatalErrorNode()) {
    // FIXME: improve grammar in the following strings:
    constexpr llvm::StringLiteral Msg =
        "Using a fixed address is not portable because that address will "
        "probably not be valid in all environments or platforms.";
    auto R = std::make_unique<PathSensitiveBugReport>(BT, Msg, N);
    R->addRange(SrcExpr->getSourceRange());
    C.emitReport(std::move(R));
  }
}

void FixedAddressChecker::checkPreStmt(const BinaryOperator *BO,
                                       CheckerContext &C) const {
  if (BO->getOpcode() != BO_Assign)
    return;

  checkUseOfFixedAddress(BO->getType(), BO->getRHS(), C);
}

void FixedAddressChecker::checkPreStmt(const DeclStmt *DS,
                                       CheckerContext &C) const {
  for (const auto *D : DS->decls()) {
    if (const auto *VD = dyn_cast<VarDecl>(D); VD && VD->hasInit())
      checkUseOfFixedAddress(VD->getType(), VD->getInit(), C);
  }
}

void FixedAddressChecker::checkPreCall(const CallEvent &Call,
                                       CheckerContext &C) const {
  for (auto Parm : enumerate(Call.parameters()))
    checkUseOfFixedAddress(Parm.value()->getType(),
                           Call.getArgExpr(Parm.index()), C);
}

void ento::registerFixedAddressChecker(CheckerManager &mgr) {
  mgr.registerChecker<FixedAddressChecker>();
}

bool ento::shouldRegisterFixedAddressChecker(const CheckerManager &mgr) {
  return true;
}
