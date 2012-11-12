//=== NoReturnFunctionChecker.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines NoReturnFunctionChecker, which evaluates functions that do not
// return to the caller.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/ADT/StringSwitch.h"
#include <cstdarg>

using namespace clang;
using namespace ento;

namespace {

class NoReturnFunctionChecker : public Checker< check::PostStmt<CallExpr>,
                                                check::PostObjCMessage > {
public:
  void checkPostStmt(const CallExpr *CE, CheckerContext &C) const;
  void checkPostObjCMessage(const ObjCMethodCall &msg, CheckerContext &C) const;
};

}

void NoReturnFunctionChecker::checkPostStmt(const CallExpr *CE,
                                            CheckerContext &C) const {
  ProgramStateRef state = C.getState();
  const Expr *Callee = CE->getCallee();

  bool BuildSinks = getFunctionExtInfo(Callee->getType()).getNoReturn();

  if (!BuildSinks) {
    SVal L = state->getSVal(Callee, C.getLocationContext());
    const FunctionDecl *FD = L.getAsFunctionDecl();
    if (!FD)
      return;

    if (FD->getAttr<AnalyzerNoReturnAttr>())
      BuildSinks = true;
    else if (const IdentifierInfo *II = FD->getIdentifier()) {
      // HACK: Some functions are not marked noreturn, and don't return.
      //  Here are a few hardwired ones.  If this takes too long, we can
      //  potentially cache these results.
      BuildSinks
        = llvm::StringSwitch<bool>(StringRef(II->getName()))
            .Case("exit", true)
            .Case("panic", true)
            .Case("error", true)
            .Case("Assert", true)
            // FIXME: This is just a wrapper around throwing an exception.
            //  Eventually inter-procedural analysis should handle this easily.
            .Case("ziperr", true)
            .Case("assfail", true)
            .Case("db_error", true)
            .Case("__assert", true)
            .Case("__assert_rtn", true)
            .Case("__assert_fail", true)
            .Case("dtrace_assfail", true)
            .Case("yy_fatal_error", true)
            .Case("_XCAssertionFailureHandler", true)
            .Case("_DTAssertionFailureHandler", true)
            .Case("_TSAssertionFailureHandler", true)
            .Default(false);
    }
  }

  if (BuildSinks)
    C.generateSink();
}

static bool END_WITH_NULL isMultiArgSelector(const Selector *Sel, ...) {
  va_list argp;
  va_start(argp, Sel);

  unsigned Slot = 0;
  const char *Arg;
  while ((Arg = va_arg(argp, const char *))) {
    if (!Sel->getNameForSlot(Slot).equals(Arg))
      break; // still need to va_end!
    ++Slot;
  }

  va_end(argp);

  // We only succeeded if we made it to the end of the argument list.
  return (Arg == NULL);
}

void NoReturnFunctionChecker::checkPostObjCMessage(const ObjCMethodCall &Msg,
                                                   CheckerContext &C) const {
  // HACK: This entire check is to handle two messages in the Cocoa frameworks:
  // -[NSAssertionHandler
  //    handleFailureInMethod:object:file:lineNumber:description:]
  // -[NSAssertionHandler
  //    handleFailureInFunction:file:lineNumber:description:]
  // Eventually these should be annotated with __attribute__((noreturn)).
  // Because ObjC messages use dynamic dispatch, it is not generally safe to
  // assume certain methods can't return. In cases where it is definitely valid,
  // see if you can mark the methods noreturn or analyzer_noreturn instead of
  // adding more explicit checks to this method.

  if (!Msg.isInstanceMessage())
    return;

  const ObjCInterfaceDecl *Receiver = Msg.getReceiverInterface();
  if (!Receiver)
    return;
  if (!Receiver->getIdentifier()->isStr("NSAssertionHandler"))
    return;

  Selector Sel = Msg.getSelector();
  switch (Sel.getNumArgs()) {
  default:
    return;
  case 4:
    if (!isMultiArgSelector(&Sel, "handleFailureInFunction", "file",
                            "lineNumber", "description", NULL))
      return;
    break;
  case 5:
    if (!isMultiArgSelector(&Sel, "handleFailureInMethod", "object", "file",
                            "lineNumber", "description", NULL))
      return;
    break;
  }

  // If we got here, it's one of the messages we care about.
  C.generateSink();
}


void ento::registerNoReturnFunctionChecker(CheckerManager &mgr) {
  mgr.registerChecker<NoReturnFunctionChecker>();
}
