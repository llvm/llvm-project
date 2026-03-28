//== UnconditionalVAArgChecker.cpp -----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines a checker to detect functions that unconditionally call
// va_arg() and would fail if they were called with zero variadic arguments.
//
// This checker is only partially path-sensitive: it relies on the symbolic
// execution of the analyzer engine to follow the execution path from the
// beginning of a function to a va_arg() call and determine whether there are
// any branching points on that path -- but it uses BasicBugReport reports
// because report path wouldn't contain any useful information. (As this
// checker diagnoses a property of a variadic function, the path before that
// function is irrelevant; then the unconditional part of path is trivial.)
//
// The AST matching framework of Clang Tidy is not powerful enough to express
// this "no branching on the execution path" relationship, at least not without
// reimplementing a crude and fragile form of symbolic execution.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/Support/FormatVariadic.h"

using namespace clang;
using namespace ento;
using llvm::formatv;

/// Either nullptr or a function under symbolic execution; a non-null value
/// means that the analyzer didn't see any branching points from the beginning
/// of that function until the current location.
REGISTER_TRAIT_WITH_PROGRAMSTATE(HasUnconditionalPath, const FunctionDecl *)

namespace {
class UnconditionalVAArgChecker
    : public Checker<check::BeginFunction, check::EndFunction,
                     check::BranchCondition, check::PreStmt<VAArgExpr>> {
  const BugType BT{this, "Unconditional use of va_arg()",
                   categories::MemoryError};

  static const FunctionDecl *getCurrentFunction(CheckerContext &C);

public:
  void checkBeginFunction(CheckerContext &C) const;
  void checkEndFunction(const ReturnStmt *, CheckerContext &C) const;
  void checkBranchCondition(const Stmt *Condition, CheckerContext &C) const;
  void checkPreStmt(const VAArgExpr *VAA, CheckerContext &C) const;
};
} // end anonymous namespace

const FunctionDecl *
UnconditionalVAArgChecker::getCurrentFunction(CheckerContext &C) {
  const Decl *FD = C.getLocationContext()->getDecl();
  return dyn_cast_if_present<FunctionDecl>(FD);
}

void UnconditionalVAArgChecker::checkBeginFunction(CheckerContext &C) const {
  // We only look for unconditional va_arg() use in variadic functions.
  // Functions that take a va_list argument are just parts of the argument
  // handling, it is more natural for them to have preconditions.
  const FunctionDecl *FD = getCurrentFunction(C);
  if (FD && FD->isVariadic()) {
    // If a variadic function is inlined in the body of another variadic
    // function, this overwrites the path tracking for the outer function. As
    // this situation is fairly rare and it is very unlikely that the "big"
    // outer function still has an unconditional path, there is no need to
    // write more complex logic that handles this.
    // NOTE: Despite this, the checker can sometimes still report the
    // unconditional va_arg() use in the outer function (probably because there
    // is an alternative execution path that doesn't enter the inner call).
    C.addTransition(C.getState()->set<HasUnconditionalPath>(FD));
  }
}

void UnconditionalVAArgChecker::checkEndFunction(const ReturnStmt *,
                                                 CheckerContext &C) const {
  // This callback is just for the sake of cleanliness, to remove data from the
  // State after it becomes irrelevant. This checker would function perfectly
  // correctly without this callback, and the impact on other checkers is also
  // extremely limited (presence of extra metadata might prevent the
  // unification of execution paths in some very rare situations).
  ProgramStateRef State = C.getState();
  const FunctionDecl *FD = getCurrentFunction(C);
  if (FD && FD == State->get<HasUnconditionalPath>()) {
    State = State->set<HasUnconditionalPath>(nullptr);
    C.addTransition(State);
  }
}

void UnconditionalVAArgChecker::checkBranchCondition(const Stmt *Condition,
                                                     CheckerContext &C) const {
  // After evaluating a branch condition, the analyzer (which examines
  // execution paths individually) won't be able to find a va_arg() expression
  // that is _unconditionally_ reached -- so this callback resets the state
  // trait HasUnconditionalPath.
  // NOTES:
  // 1. This is the right thing to do even if the analyzer sees that _in the
  // current state_ the execution can only continue in one direction. For
  // example, if the variadic function isn't the entrypont, then the parameters
  // recieved from the caller may guarantee that va_arg() is used -- but this
  // does not mean that the function _unconditionally_ uses va_arg().
  // 2. After other kinds of state splits (e.g. EagerlyAssueme, callbacks of
  // StdLibraryFunctions separating different cases for the behavior of a
  // library function etc.) the different execution paths will follow the same
  // code (until they hit a branch condition), so it is reasonable (although
  // not always correct) to assume that a va_arg() reached after those state
  // splits is still _unconditionally_ reached if there were no branching
  // statements.
  // 3. This checker activates _after_ the evaluation of the branch condition,
  // so va_arg() in the branch condition can be unconditionally reached.
  C.addTransition(C.getState()->set<HasUnconditionalPath>(nullptr));
}

void UnconditionalVAArgChecker::checkPreStmt(const VAArgExpr *VAA,
                                             CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  const FunctionDecl *PathFrom = State->get<HasUnconditionalPath>();
  if (!PathFrom)
    return;

  // Reset this trait in the state to ensure that multiple consecutive
  // va_arg() calls don't produce repeated warnings.
  C.addTransition(State->set<HasUnconditionalPath>(nullptr));

  IdentifierInfo *II = PathFrom->getIdentifier();
  if (!II)
    return;
  StringRef FN = II->getName();

  std::string FullMsg = formatv(
      "Calls to '{0}' always reach this va_arg() expression, so calling "
      "'{0}' with no variadic arguments would be undefined behavior",
      FN);
  SourceRange SR = VAA->getSourceRange();
  PathDiagnosticLocation PDL(SR.getBegin(),
                             C.getASTContext().getSourceManager());
  // We create a non-path-sensitive report because the path wouldn't contain
  // any useful information: the path leading to the variadic function is
  // actively ignored by the checker and the unconditional path from the
  // start of the variadic function is trivial.
  auto R =
      std::make_unique<BasicBugReport>(BT, BT.getDescription(), FullMsg, PDL);

  if (getCurrentFunction(C) != PathFrom) {
    // Highlight the definition of the variadic function in the rare case
    // when the reached va_arg() expression is in another function.
    SourceRange DefSR = PathFrom->getSourceRange();
    PathDiagnosticLocation DefPDL(DefSR.getBegin(),
                                  C.getASTContext().getSourceManager());
    std::string NoteMsg =
        formatv("Variadic function '{0}' is defined here", FN);
    R->addNote(NoteMsg, DefPDL, DefSR);
  }
  R->addRange(SR);
  R->setDeclWithIssue(PathFrom);
  C.emitReport(std::move(R));
}

void ento::registerUnconditionalVAArgChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<UnconditionalVAArgChecker>();
}

bool ento::shouldRegisterUnconditionalVAArgChecker(const CheckerManager &) {
  return true;
}
