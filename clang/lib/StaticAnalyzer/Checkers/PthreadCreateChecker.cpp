//
// Created by MaxSa on 11/13/2024.
//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

class PthreadCreateChecker : public Checker<check::PostCall> {
public:
  void checkPostCall(const CallEvent &Call, CheckerContext &Context) const;

};

void PthreadCreateChecker::checkPostCall(const CallEvent &Call, CheckerContext &Context) const {
  const FunctionDecl *FuncID = Call.getDecl()->getAsFunction();
  if (!FuncID) {
    return;
  }

  if (FuncID->getName() == "pthread_create") {
    SVal returnVal = Call.getReturnValue();
    if (returnVal.isZeroConstant()) {
      llvm::errs() << "Pthread has been created\n";
    }
  }
}

// Register checker
void ento::registerPthreadCreateChecker(CheckerManager &mgr) {
  mgr.registerChecker<PthreadCreateChecker>();
}

bool ento::shouldRegisterPthreadCreateChecker(const CheckerManager &mgr) {
  return true;
}



