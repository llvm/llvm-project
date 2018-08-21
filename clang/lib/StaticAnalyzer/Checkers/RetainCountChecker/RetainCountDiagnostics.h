//== RetainCountDiagnostics.h - Checks for leaks and other issues -*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines diagnostics for RetainCountChecker, which implements
//  a reference count checker for Core Foundation and Cocoa on (Mac OS X).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_RETAINCOUNTCHECKER_DIAGNOSTICS_H
#define LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_RETAINCOUNTCHECKER_DIAGNOSTICS_H

#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporterVisitors.h"
#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Core/RetainSummaryManager.h"

namespace clang {
namespace ento {
namespace retaincountchecker {

class CFRefBug : public BugType {
protected:
  CFRefBug(const CheckerBase *checker, StringRef name)
      : BugType(checker, name, categories::MemoryCoreFoundationObjectiveC) {}

public:

  // FIXME: Eventually remove.
  virtual const char *getDescription() const = 0;

  virtual bool isLeak() const { return false; }
};

class UseAfterRelease : public CFRefBug {
public:
  UseAfterRelease(const CheckerBase *checker)
      : CFRefBug(checker, "Use-after-release") {}

  const char *getDescription() const override {
    return "Reference-counted object is used after it is released";
  }
};

class BadRelease : public CFRefBug {
public:
  BadRelease(const CheckerBase *checker) : CFRefBug(checker, "Bad release") {}

  const char *getDescription() const override {
    return "Incorrect decrement of the reference count of an object that is "
           "not owned at this point by the caller";
  }
};

class DeallocNotOwned : public CFRefBug {
public:
  DeallocNotOwned(const CheckerBase *checker)
      : CFRefBug(checker, "-dealloc sent to non-exclusively owned object") {}

  const char *getDescription() const override {
    return "-dealloc sent to object that may be referenced elsewhere";
  }
};

class OverAutorelease : public CFRefBug {
public:
  OverAutorelease(const CheckerBase *checker)
      : CFRefBug(checker, "Object autoreleased too many times") {}

  const char *getDescription() const override {
    return "Object autoreleased too many times";
  }
};

class ReturnedNotOwnedForOwned : public CFRefBug {
public:
  ReturnedNotOwnedForOwned(const CheckerBase *checker)
      : CFRefBug(checker, "Method should return an owned object") {}

  const char *getDescription() const override {
    return "Object with a +0 retain count returned to caller where a +1 "
           "(owning) retain count is expected";
  }
};

class Leak : public CFRefBug {
public:
  Leak(const CheckerBase *checker, StringRef name) : CFRefBug(checker, name) {
    // Leaks should not be reported if they are post-dominated by a sink.
    setSuppressOnSink(true);
  }

  const char *getDescription() const override { return ""; }

  bool isLeak() const override { return true; }
};

typedef ::llvm::DenseMap<const ExplodedNode *, const RetainSummary *>
  SummaryLogTy;

/// Visitors.

class CFRefReportVisitor : public BugReporterVisitor {
protected:
  SymbolRef Sym;
  const SummaryLogTy &SummaryLog;

public:
  CFRefReportVisitor(SymbolRef sym, const SummaryLogTy &log)
     : Sym(sym), SummaryLog(log) {}

  void Profile(llvm::FoldingSetNodeID &ID) const override {
    static int x = 0;
    ID.AddPointer(&x);
    ID.AddPointer(Sym);
  }

  std::shared_ptr<PathDiagnosticPiece> VisitNode(const ExplodedNode *N,
                                                 const ExplodedNode *PrevN,
                                                 BugReporterContext &BRC,
                                                 BugReport &BR) override;

  std::shared_ptr<PathDiagnosticPiece> getEndPath(BugReporterContext &BRC,
                                                  const ExplodedNode *N,
                                                  BugReport &BR) override;
};

class CFRefLeakReportVisitor : public CFRefReportVisitor {
public:
  CFRefLeakReportVisitor(SymbolRef sym,
                         const SummaryLogTy &log)
     : CFRefReportVisitor(sym, log) {}

  std::shared_ptr<PathDiagnosticPiece> getEndPath(BugReporterContext &BRC,
                                                  const ExplodedNode *N,
                                                  BugReport &BR) override;
};

class CFRefReport : public BugReport {

public:
  CFRefReport(CFRefBug &D, const LangOptions &LOpts,
              const SummaryLogTy &Log, ExplodedNode *n, SymbolRef sym,
              bool registerVisitor = true)
    : BugReport(D, D.getDescription(), n) {
    if (registerVisitor)
      addVisitor(llvm::make_unique<CFRefReportVisitor>(sym, Log));
  }

  CFRefReport(CFRefBug &D, const LangOptions &LOpts,
              const SummaryLogTy &Log, ExplodedNode *n, SymbolRef sym,
              StringRef endText)
    : BugReport(D, D.getDescription(), endText, n) {
    addVisitor(llvm::make_unique<CFRefReportVisitor>(sym, Log));
  }

  llvm::iterator_range<ranges_iterator> getRanges() override {
    const CFRefBug& BugTy = static_cast<CFRefBug&>(getBugType());
    if (!BugTy.isLeak())
      return BugReport::getRanges();
    return llvm::make_range(ranges_iterator(), ranges_iterator());
  }
};

class CFRefLeakReport : public CFRefReport {
  const MemRegion* AllocBinding;
  const Stmt *AllocStmt;

  // Finds the function declaration where a leak warning for the parameter
  // 'sym' should be raised.
  void deriveParamLocation(CheckerContext &Ctx, SymbolRef sym);
  // Finds the location where a leak warning for 'sym' should be raised.
  void deriveAllocLocation(CheckerContext &Ctx, SymbolRef sym);
  // Produces description of a leak warning which is printed on the console.
  void createDescription(CheckerContext &Ctx, bool IncludeAllocationLine);

public:
  CFRefLeakReport(CFRefBug &D, const LangOptions &LOpts,
                  const SummaryLogTy &Log, ExplodedNode *n, SymbolRef sym,
                  CheckerContext &Ctx,
                  bool IncludeAllocationLine);

  PathDiagnosticLocation getLocation(const SourceManager &SM) const override {
    assert(Location.isValid());
    return Location;
  }
};

} // end namespace retaincountchecker
} // end namespace ento
} // end namespace clang

#endif
