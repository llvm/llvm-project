//=== CastSizeChecker.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// CastSizeChecker checks when casting a malloc'ed symbolic region to type T,
// whether the size of the symbolic region is a multiple of the size of T.
//
//===----------------------------------------------------------------------===//
#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/AST/CharUnits.h"

using namespace clang;
using namespace ento;

namespace {
class CastSizeChecker : public Checker< check::PreStmt<CastExpr> > {
  mutable OwningPtr<BuiltinBug> BT;
public:
  void checkPreStmt(const CastExpr *CE, CheckerContext &C) const;
};
}

void CastSizeChecker::checkPreStmt(const CastExpr *CE,CheckerContext &C) const {
  const Expr *E = CE->getSubExpr();
  ASTContext &Ctx = C.getASTContext();
  QualType ToTy = Ctx.getCanonicalType(CE->getType());
  const PointerType *ToPTy = dyn_cast<PointerType>(ToTy.getTypePtr());

  if (!ToPTy)
    return;

  QualType ToPointeeTy = ToPTy->getPointeeType();

  // Only perform the check if 'ToPointeeTy' is a complete type.
  if (ToPointeeTy->isIncompleteType())
    return;

  ProgramStateRef state = C.getState();
  const MemRegion *R = state->getSVal(E, C.getLocationContext()).getAsRegion();
  if (R == 0)
    return;

  const SymbolicRegion *SR = dyn_cast<SymbolicRegion>(R);
  if (SR == 0)
    return;

  SValBuilder &svalBuilder = C.getSValBuilder();
  SVal extent = SR->getExtent(svalBuilder);
  const llvm::APSInt *extentInt = svalBuilder.getKnownValue(state, extent);
  if (!extentInt)
    return;

  CharUnits regionSize = CharUnits::fromQuantity(extentInt->getSExtValue());
  CharUnits typeSize = C.getASTContext().getTypeSizeInChars(ToPointeeTy);

  // Ignore void, and a few other un-sizeable types.
  if (typeSize.isZero())
    return;

  if (regionSize % typeSize != 0) {
    if (ExplodedNode *errorNode = C.generateSink()) {
      if (!BT)
        BT.reset(new BuiltinBug("Cast region with wrong size.",
                            "Cast a region whose size is not a multiple of the"
                            " destination type size."));
      BugReport *R = new BugReport(*BT, BT->getDescription(),
                                               errorNode);
      R->addRange(CE->getSourceRange());
      C.emitReport(R);
    }
  }
}


void ento::registerCastSizeChecker(CheckerManager &mgr) {
  mgr.registerChecker<CastSizeChecker>();  
}
