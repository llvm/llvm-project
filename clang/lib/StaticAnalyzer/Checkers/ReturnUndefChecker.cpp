//== ReturnUndefChecker.cpp -------------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines ReturnUndefChecker, which is a path-sensitive
// check which looks for undefined or garbage values being returned to the
// caller.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"

using namespace clang;
using namespace ento;

namespace {
class ReturnUndefChecker : 
    public Checker< check::PreStmt<ReturnStmt> > {
  mutable OwningPtr<BuiltinBug> BT;
public:
  void checkPreStmt(const ReturnStmt *RS, CheckerContext &C) const;
};
}

void ReturnUndefChecker::checkPreStmt(const ReturnStmt *RS,
                                      CheckerContext &C) const {
 
  const Expr *RetE = RS->getRetValue();
  if (!RetE)
    return;
  
  if (!C.getState()->getSVal(RetE, C.getLocationContext()).isUndef())
    return;
  
  // "return;" is modeled to evaluate to an UndefinedValue. Allow UndefinedValue
  // to be returned in functions returning void to support the following pattern:
  // void foo() {
  //  return;
  // }
  // void test() {
  //   return foo();
  // }
  const StackFrameContext *SFC = C.getStackFrame();
  QualType RT = CallEvent::getDeclaredResultType(SFC->getDecl());
  if (!RT.isNull() && RT->isSpecificBuiltinType(BuiltinType::Void))
    return;

  ExplodedNode *N = C.generateSink();

  if (!N)
    return;
  
  if (!BT)
    BT.reset(new BuiltinBug("Garbage return value",
                            "Undefined or garbage value returned to caller"));
    
  BugReport *report = 
    new BugReport(*BT, BT->getDescription(), N);

  report->addRange(RetE->getSourceRange());
  bugreporter::trackNullOrUndefValue(N, RetE, *report);

  C.emitReport(report);
}

void ento::registerReturnUndefChecker(CheckerManager &mgr) {
  mgr.registerChecker<ReturnUndefChecker>();
}
