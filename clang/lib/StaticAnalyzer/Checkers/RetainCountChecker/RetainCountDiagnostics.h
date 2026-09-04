//== RetainCountDiagnostics.h - Checks for leaks and other issues -*- C++ -*--//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines diagnostics for RetainCountChecker, which implements
//  a reference count checker for Core Foundation and Cocoa on (Mac OS X).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_RETAINCOUNTCHECKER_DIAGNOSTICS_H
#define LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_RETAINCOUNTCHECKER_DIAGNOSTICS_H

#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Analysis/RetainSummaryManager.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporterVisitors.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"

namespace clang {
namespace ento {
namespace retaincountchecker {

class RefCountBug : public BugType {
  StringRef ReportMessage;

public:
  RefCountBug(const CheckerFrontend *CF, StringRef Desc, StringRef ReportMsg,
              bool SuppressOnSink = false)
      : BugType(CF, Desc, categories::MemoryRefCount, SuppressOnSink),
        ReportMessage(ReportMsg) {}
  StringRef getReportMessage() const { return ReportMessage; }
};

class RefCountFrontend : public CheckerFrontend {
public:
  const RefCountBug UseAfterRelease{
      this, "Use-after-release",
      "Reference-counted object is used after it is released"};
  const RefCountBug ReleaseNotOwned{
      this, "Bad release",
      "Incorrect decrement of the reference count of an object that is not "
      "owned at this point by the caller"};
  const RefCountBug DeallocNotOwned{
      this, "-dealloc sent to non-exclusively owned object",
      "-dealloc sent to object that may be referenced elsewhere"};
  const RefCountBug FreeNotOwned{
      this, "freeing non-exclusively owned object",
      "'free' called on an object that may be referenced elsewhere"};
  const RefCountBug OverAutorelease{this, "Object autoreleased too many times",
                                    "Object autoreleased too many times"};
  const RefCountBug ReturnNotOwnedForOwned{
      this, "Method should return an owned object",
      "Object with a +0 retain count returned to caller where a +1 (owning) "
      "retain count is expected"};
  // For these two bug types the report message will be generated dynamically
  // by `RefLeakReport::createDescription` so the empty string taken from the
  // BugType will be ignored (overwritten).
  const RefCountBug LeakWithinFunction{this, "Leak", /*ReportMsg=*/"",
                                       /*SuppressOnSink=*/true};
  const RefCountBug LeakAtReturn{this, "Leak of returned object",
                                 /*ReportMsg=*/"", /*SuppressOnSink=*/true};
};

class RefCountReport : public PathSensitiveBugReport {
protected:
  SymbolRef Sym;
  bool isLeak = false;

public:
  RefCountReport(const RefCountBug &D, const LangOptions &LOpts,
                 ExplodedNode *n, SymbolRef sym, bool isLeak = false,
                 bool IsReleaseUnowned = false);

  RefCountReport(const RefCountBug &D, const LangOptions &LOpts,
              ExplodedNode *n, SymbolRef sym,
              StringRef endText);

  ArrayRef<SourceRange> getRanges() const override {
    if (!isLeak)
      return PathSensitiveBugReport::getRanges();
    return {};
  }
};

class RefLeakReport : public RefCountReport {
  const MemRegion *AllocFirstBinding = nullptr;
  const MemRegion *AllocBindingToReport = nullptr;
  const Stmt *AllocStmt = nullptr;
  PathDiagnosticLocation Location;

  // Finds the function declaration where a leak warning for the parameter
  // 'sym' should be raised.
  void deriveParamLocation(CheckerContext &Ctx);
  // Finds the location where the leaking object is allocated.
  void deriveAllocLocation(CheckerContext &Ctx);
  // Produces description of a leak warning which is printed on the console.
  void createDescription(CheckerContext &Ctx);
  // Finds the binding that we should use in a leak warning.
  void findBindingToReport(CheckerContext &Ctx, ExplodedNode *Node);

public:
  RefLeakReport(const RefCountBug &D, const LangOptions &LOpts, ExplodedNode *n,
                SymbolRef sym, CheckerContext &Ctx);
  PathDiagnosticLocation getLocation() const override {
    assert(Location.isValid());
    return Location;
  }

  PathDiagnosticLocation getEndOfPath() const {
    return PathSensitiveBugReport::getLocation();
  }
};

} // end namespace retaincountchecker
} // end namespace ento
} // end namespace clang

#endif
