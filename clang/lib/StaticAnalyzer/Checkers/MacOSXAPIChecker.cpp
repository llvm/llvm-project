// MacOSXAPIChecker.h - Checks proper use of various MacOS X APIs --*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines MacOSXAPIChecker, which is an assortment of checks on calls
// to various, widely used Apple APIs.
//
// FIXME: What's currently in BasicObjCFoundationChecks.cpp should be migrated
// to here, using the new Checker interface.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

namespace {
class MacOSXAPIChecker : public Checker< check::PreStmt<CallExpr> > {
  mutable std::unique_ptr<BugType> BT_dispatchOnce;

public:
  void checkPreStmt(const CallExpr *CE, CheckerContext &C) const;

  void CheckDispatchOnce(CheckerContext &C, const CallExpr *CE,
                         StringRef FName) const;

  typedef void (MacOSXAPIChecker::*SubChecker)(CheckerContext &,
                                               const CallExpr *,
                                               StringRef FName) const;
};
} //end anonymous namespace

//===----------------------------------------------------------------------===//
// dispatch_once and dispatch_once_f
//===----------------------------------------------------------------------===//

void MacOSXAPIChecker::CheckDispatchOnce(CheckerContext &C, const CallExpr *CE,
                                         StringRef FName) const {
  if (CE->getNumArgs() < 1)
    return;

  // Check if the first argument is stack allocated.  If so, issue a warning
  // because that's likely to be bad news.
  ProgramStateRef state = C.getState();
  const MemRegion *R =
    state->getSVal(CE->getArg(0), C.getLocationContext()).getAsRegion();
  if (!R || !isa<StackSpaceRegion>(R->getMemorySpace()))
    return;

  ExplodedNode *N = C.generateErrorNode(state);
  if (!N)
    return;

  if (!BT_dispatchOnce)
    BT_dispatchOnce.reset(new BugType(this, "Improper use of 'dispatch_once'",
                                      "API Misuse (Apple)"));

  // Handle _dispatch_once.  In some versions of the OS X SDK we have the case
  // that dispatch_once is a macro that wraps a call to _dispatch_once.
  // _dispatch_once is then a function which then calls the real dispatch_once.
  // Users do not care; they just want the warning at the top-level call.
  if (CE->getLocStart().isMacroID()) {
    StringRef TrimmedFName = FName.ltrim('_');
    if (TrimmedFName != FName)
      FName = TrimmedFName;
  }

  SmallString<256> S;
  llvm::raw_svector_ostream os(S);
  os << "Call to '" << FName << "' uses";
  if (const VarRegion *VR = dyn_cast<VarRegion>(R))
    os << " the local variable '" << VR->getDecl()->getName() << '\'';
  else
    os << " stack allocated memory";
  os << " for the predicate value.  Using such transient memory for "
        "the predicate is potentially dangerous.";
  if (isa<VarRegion>(R) && isa<StackLocalsSpaceRegion>(R->getMemorySpace()))
    os << "  Perhaps you intended to declare the variable as 'static'?";

  auto report = llvm::make_unique<BugReport>(*BT_dispatchOnce, os.str(), N);
  report->addRange(CE->getArg(0)->getSourceRange());
  C.emitReport(std::move(report));
}

//===----------------------------------------------------------------------===//
// Central dispatch function.
//===----------------------------------------------------------------------===//

void MacOSXAPIChecker::checkPreStmt(const CallExpr *CE,
                                    CheckerContext &C) const {
  StringRef Name = C.getCalleeName(CE);
  if (Name.empty())
    return;

  SubChecker SC =
    llvm::StringSwitch<SubChecker>(Name)
      .Cases("dispatch_once",
             "_dispatch_once",
             "dispatch_once_f",
             &MacOSXAPIChecker::CheckDispatchOnce)
      .Default(nullptr);

  if (SC)
    (this->*SC)(C, CE, Name);
}

//===----------------------------------------------------------------------===//
// Registration.
//===----------------------------------------------------------------------===//

void ento::registerMacOSXAPIChecker(CheckerManager &mgr) {
  mgr.registerChecker<MacOSXAPIChecker>();
}
