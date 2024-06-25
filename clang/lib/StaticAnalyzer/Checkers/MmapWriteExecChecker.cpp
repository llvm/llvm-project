// MmapWriteExecChecker.cpp - Check for the prot argument -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This checker tests the 3rd argument of mmap's calls to check if
// it is writable and executable in the same time. It's somehow
// an optional checker since for example in JIT libraries it is pretty common.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"

#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/BugReporter/CommonBugCategories.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerHelpers.h"

using namespace clang;
using namespace ento;

namespace {
class MmapWriteExecChecker
    : public Checker<check::BeginFunction, check::PreCall> {
  CallDescription MmapFn{CDM::CLibrary, {"mmap"}, 6};
  CallDescription MprotectFn{CDM::CLibrary, {"mprotect"}, 3};
  const BugType BT{this, "W^X check fails, Write Exec prot flags set",
                   "Security"};

  mutable bool FlagsInitialized = false;
  mutable int ProtRead = 0x01;
  mutable int ProtWrite = 0x02;
  mutable int ProtExec = 0x04;

public:
  void checkBeginFunction(CheckerContext &C) const;
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;

  int ProtExecOv;
  int ProtReadOv;
};
}

void MmapWriteExecChecker::checkBeginFunction(CheckerContext &C) const {
  if (FlagsInitialized)
    return;

  FlagsInitialized = true;

  const std::optional<int> FoundProtRead =
          tryExpandAsInteger("PROT_READ", C.getPreprocessor());
  const std::optional<int> FoundProtWrite =
          tryExpandAsInteger("PROT_WRITE", C.getPreprocessor());
  const std::optional<int> FoundProtExec =
          tryExpandAsInteger("PROT_EXEC", C.getPreprocessor());
  if (FoundProtRead && FoundProtWrite && FoundProtExec) {
    ProtRead = *FoundProtRead;
    ProtWrite = *FoundProtWrite;
    ProtExec = *FoundProtExec;
  } else {
    // FIXME: Are these useful?
    ProtRead = ProtReadOv;
    ProtExec = ProtExecOv;
  }
}

void MmapWriteExecChecker::checkPreCall(const CallEvent &Call,
                                         CheckerContext &C) const {
  if (matchesAny(Call, MmapFn, MprotectFn)) {
    SVal ProtVal = Call.getArgSVal(2);
    auto ProtLoc = ProtVal.getAs<nonloc::ConcreteInt>();
    if (!ProtLoc)
      return;
    int64_t Prot = ProtLoc->getValue().getSExtValue();

    if ((Prot & ProtWrite) && (Prot & ProtExec)) {
      ExplodedNode *N = C.generateNonFatalErrorNode();
      if (!N)
        return;

      auto Report = std::make_unique<PathSensitiveBugReport>(
          BT,
          "Both PROT_WRITE and PROT_EXEC flags are set. This can "
          "lead to exploitable memory regions, which could be overwritten "
          "with malicious code",
          N);
      Report->addRange(Call.getArgSourceRange(2));
      C.emitReport(std::move(Report));
    }
  }
}

void ento::registerMmapWriteExecChecker(CheckerManager &mgr) {
  MmapWriteExecChecker *Mwec =
      mgr.registerChecker<MmapWriteExecChecker>();
  Mwec->ProtExecOv =
    mgr.getAnalyzerOptions()
      .getCheckerIntegerOption(Mwec, "MmapProtExec");
  Mwec->ProtReadOv =
    mgr.getAnalyzerOptions()
      .getCheckerIntegerOption(Mwec, "MmapProtRead");
}

bool ento::shouldRegisterMmapWriteExecChecker(const CheckerManager &mgr) {
  return true;
}
