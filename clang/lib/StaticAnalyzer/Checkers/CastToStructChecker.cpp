//=== CastToStructChecker.cpp - Fixed address usage checker ----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This files defines CastToStructChecker, a builtin checker that checks for
// cast from non-struct pointer to struct pointer.
// This check corresponds to CWE-588.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"

using namespace clang;
using namespace ento;

namespace {
class CastToStructChecker : public Checker< check::PreStmt<CastExpr> > {
  mutable OwningPtr<BuiltinBug> BT;

public:
  void checkPreStmt(const CastExpr *CE, CheckerContext &C) const;
};
}

void CastToStructChecker::checkPreStmt(const CastExpr *CE,
                                       CheckerContext &C) const {
  const Expr *E = CE->getSubExpr();
  ASTContext &Ctx = C.getASTContext();
  QualType OrigTy = Ctx.getCanonicalType(E->getType());
  QualType ToTy = Ctx.getCanonicalType(CE->getType());

  const PointerType *OrigPTy = dyn_cast<PointerType>(OrigTy.getTypePtr());
  const PointerType *ToPTy = dyn_cast<PointerType>(ToTy.getTypePtr());

  if (!ToPTy || !OrigPTy)
    return;

  QualType OrigPointeeTy = OrigPTy->getPointeeType();
  QualType ToPointeeTy = ToPTy->getPointeeType();

  if (!ToPointeeTy->isStructureOrClassType())
    return;

  // We allow cast from void*.
  if (OrigPointeeTy->isVoidType())
    return;

  // Now the cast-to-type is struct pointer, the original type is not void*.
  if (!OrigPointeeTy->isRecordType()) {
    if (ExplodedNode *N = C.addTransition()) {
      if (!BT)
        BT.reset(new BuiltinBug("Cast from non-struct type to struct type",
                            "Casting a non-structure type to a structure type "
                            "and accessing a field can lead to memory access "
                            "errors or data corruption."));
      BugReport *R = new BugReport(*BT,BT->getDescription(), N);
      R->addRange(CE->getSourceRange());
      C.emitReport(R);
    }
  }
}

void ento::registerCastToStructChecker(CheckerManager &mgr) {
  mgr.registerChecker<CastToStructChecker>();
}
