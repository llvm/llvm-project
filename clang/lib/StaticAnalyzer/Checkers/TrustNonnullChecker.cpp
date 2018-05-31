//== TrustNonnullChecker.cpp - Checker for trusting annotations -*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This checker adds an assumption that methods annotated with _Nonnull
// which come from system headers actually return a non-null pointer.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerHelpers.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"

using namespace clang;
using namespace ento;

namespace {

class TrustNonnullChecker : public Checker<check::PostCall> {
private:
  /// \returns Whether we trust the result of the method call to be
  /// a non-null pointer.
  bool isNonNullPtr(const CallEvent &Call, CheckerContext &C) const {
    QualType ExprRetType = Call.getResultType();
    if (!ExprRetType->isAnyPointerType())
      return false;

    if (getNullabilityAnnotation(ExprRetType) == Nullability::Nonnull)
      return true;

    // The logic for ObjC instance method calls is more complicated,
    // as the return value is nil when the receiver is nil.
    if (!isa<ObjCMethodCall>(&Call))
      return false;

    const auto *MCall = cast<ObjCMethodCall>(&Call);
    const ObjCMethodDecl *MD = MCall->getDecl();

    // Distrust protocols.
    if (isa<ObjCProtocolDecl>(MD->getDeclContext()))
      return false;

    QualType DeclRetType = MD->getReturnType();
    if (getNullabilityAnnotation(DeclRetType) != Nullability::Nonnull)
      return false;

    // For class messages it is sufficient for the declaration to be
    // annotated _Nonnull.
    if (!MCall->isInstanceMessage())
      return true;

    // Alternatively, the analyzer could know that the receiver is not null.
    SVal Receiver = MCall->getReceiverSVal();
    ConditionTruthVal TV = C.getState()->isNonNull(Receiver);
    if (TV.isConstrainedTrue())
      return true;

    return false;
  }

public:
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const {
    // Only trust annotations for system headers for non-protocols.
    if (!Call.isInSystemHeader())
      return;

    ProgramStateRef State = C.getState();

    if (isNonNullPtr(Call, C))
      if (auto L = Call.getReturnValue().getAs<Loc>())
        State = State->assume(*L, /*Assumption=*/true);

    C.addTransition(State);
  }
};

} // end empty namespace


void ento::registerTrustNonnullChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<TrustNonnullChecker>();
}
