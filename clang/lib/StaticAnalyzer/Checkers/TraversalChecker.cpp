//== TraversalChecker.cpp -------------------------------------- -*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These checkers print various aspects of the ExprEngine's traversal of the CFG
// as it builds the ExplodedGraph.
//
//===----------------------------------------------------------------------===//
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/StmtObjC.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

namespace {
// TODO: This checker is only referenced from two small test files and it
// doesn't seem to be useful for manual debugging, so consider reimplementing
// those tests with more modern tools and removing this checker.
class TraversalDumper
    : public Checker<check::BeginFunction, check::EndFunction> {
public:
  void checkBeginFunction(CheckerContext &C) const;
  void checkEndFunction(const ReturnStmt *RS, CheckerContext &C) const;
};
}

void TraversalDumper::checkBeginFunction(CheckerContext &C) const {
  llvm::outs() << "--BEGIN FUNCTION--\n";
}

void TraversalDumper::checkEndFunction(const ReturnStmt *RS,
                                       CheckerContext &C) const {
  llvm::outs() << "--END FUNCTION--\n";
}

void ento::registerTraversalDumper(CheckerManager &mgr) {
  mgr.registerChecker<TraversalDumper>();
}

bool ento::shouldRegisterTraversalDumper(const CheckerManager &mgr) {
  return true;
}

//------------------------------------------------------------------------------

namespace {
// TODO: This checker appears to be a utility for creating `FileCheck` tests
// verifying its stdout output, but there are no tests that rely on it, so
// perhaps it should be removed.
class CallDumper : public Checker< check::PreCall,
                                   check::PostCall > {
public:
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
};
}

void CallDumper::checkPreCall(const CallEvent &Call, CheckerContext &C) const {
  unsigned Indentation = 0;
  for (const LocationContext *LC = C.getLocationContext()->getParent();
       LC != nullptr; LC = LC->getParent())
    ++Indentation;

  // It is mildly evil to print directly to llvm::outs() rather than emitting
  // warnings, but this ensures things do not get filtered out by the rest of
  // the static analyzer machinery.
  llvm::outs().indent(Indentation);
  Call.dump(llvm::outs());
}

void CallDumper::checkPostCall(const CallEvent &Call, CheckerContext &C) const {
  const Expr *CallE = Call.getOriginExpr();
  if (!CallE)
    return;

  unsigned Indentation = 0;
  for (const LocationContext *LC = C.getLocationContext()->getParent();
       LC != nullptr; LC = LC->getParent())
    ++Indentation;

  // It is mildly evil to print directly to llvm::outs() rather than emitting
  // warnings, but this ensures things do not get filtered out by the rest of
  // the static analyzer machinery.
  llvm::outs().indent(Indentation);
  if (Call.getResultType()->isVoidType())
    llvm::outs() << "Returning void\n";
  else
    llvm::outs() << "Returning " << C.getSVal(CallE) << "\n";
}

void ento::registerCallDumper(CheckerManager &mgr) {
  mgr.registerChecker<CallDumper>();
}

bool ento::shouldRegisterCallDumper(const CheckerManager &mgr) {
  return true;
}
