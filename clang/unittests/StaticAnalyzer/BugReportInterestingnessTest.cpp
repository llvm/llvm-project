//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Reusables.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ento;
using namespace llvm;

namespace {

class InterestingnessTestChecker : public Checker<check::PreCall> {
  BugType BT_TestBug;

  using HandlerFn = std::function<void(const InterestingnessTestChecker *,
                                       const CallEvent &, CheckerContext &)>;

  CallDescriptionMap<HandlerFn> Handlers = {
      {{{"setInteresting"}, 1}, &InterestingnessTestChecker::handleInteresting},
      {{{"setNotInteresting"}, 1},
       &InterestingnessTestChecker::handleNotInteresting},
      {{{"check"}, 1}, &InterestingnessTestChecker::handleCheck},
      {{{"bug"}, 1}, &InterestingnessTestChecker::handleBug},
  };

  void handleInteresting(const CallEvent &Call, CheckerContext &C) const;
  void handleNotInteresting(const CallEvent &Call, CheckerContext &C) const;
  void handleCheck(const CallEvent &Call, CheckerContext &C) const;
  void handleBug(const CallEvent &Call, CheckerContext &C) const;

public:
  InterestingnessTestChecker()
      : BT_TestBug(this, "InterestingnessTestBug", "Test") {}

  void checkPreCall(const CallEvent &Call, CheckerContext &C) const {
    const HandlerFn *Handler = Handlers.lookup(Call);
    if (!Handler)
      return;

    (*Handler)(this, Call, C);
  }
};

} // namespace

void InterestingnessTestChecker::handleInteresting(const CallEvent &Call,
                                                   CheckerContext &C) const {
  SymbolRef Sym = Call.getArgSVal(0).getAsSymbol();
  assert(Sym);
  C.addTransition(nullptr, C.getNoteTag([Sym](PathSensitiveBugReport &BR) {
    BR.markInteresting(Sym);
    return "";
  }));
}

void InterestingnessTestChecker::handleNotInteresting(const CallEvent &Call,
                                                      CheckerContext &C) const {
  SymbolRef Sym = Call.getArgSVal(0).getAsSymbol();
  assert(Sym);
  C.addTransition(nullptr, C.getNoteTag([Sym](PathSensitiveBugReport &BR) {
    BR.markNotInteresting(Sym);
    return "";
  }));
}

void InterestingnessTestChecker::handleCheck(const CallEvent &Call,
                                             CheckerContext &C) const {
  SymbolRef Sym = Call.getArgSVal(0).getAsSymbol();
  assert(Sym);
  C.addTransition(nullptr, C.getNoteTag([Sym](PathSensitiveBugReport &BR) {
    if (BR.isInteresting(Sym))
      return "Interesting";
    else
      return "NotInteresting";
  }));
}

void InterestingnessTestChecker::handleBug(const CallEvent &Call,
                                           CheckerContext &C) const {
  ExplodedNode *N = C.generateErrorNode();
  C.emitReport(
      std::make_unique<PathSensitiveBugReport>(BT_TestBug, "test bug", N));
}

namespace {

class TestAction : public ASTFrontendAction {
  ExpectedDiagsTy ExpectedDiags;

public:
  TestAction(ExpectedDiagsTy &&ExpectedDiags)
      : ExpectedDiags(std::move(ExpectedDiags)) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &Compiler,
                                                 StringRef File) override {
    std::unique_ptr<AnalysisASTConsumer> AnalysisConsumer =
        CreateAnalysisConsumer(Compiler);
    AnalysisConsumer->AddDiagnosticConsumer(new VerifyPathDiagnosticConsumer(
        std::move(ExpectedDiags), Compiler.getSourceManager()));
    AnalysisConsumer->AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
      Registry.addChecker<InterestingnessTestChecker>("test.Interestingness",
                                                      "Description", "");
    });
    Compiler.getAnalyzerOpts()->CheckersAndPackages = {
        {"test.Interestingness", true}};
    return std::move(AnalysisConsumer);
  }
};

} // namespace

TEST(BugReportInterestingness, Symbols) {
  EXPECT_TRUE(tooling::runToolOnCode(
      std::make_unique<TestAction>(ExpectedDiagsTy{
          {{15, 7},
           "test bug",
           "test bug",
           "test.Interestingness",
           "InterestingnessTestBug",
           "Test",
           {
               {{8, 7}, "Interesting", {{{8, 7}, {8, 14}}}},
               {{10, 7}, "NotInteresting", {{{10, 7}, {10, 14}}}},
               {{12, 7}, "Interesting", {{{12, 7}, {12, 14}}}},
               {{14, 7}, "NotInteresting", {{{14, 7}, {14, 14}}}},
               {{15, 7}, "test bug", {{{15, 7}, {15, 12}}}},
           }}}),
      R"(
    void setInteresting(int);
    void setNotInteresting(int);
    void check(int);
    void bug(int);

    void f(int A) {
      check(A);
      setInteresting(A);
      check(A);
      setNotInteresting(A);
      check(A);
      setInteresting(A);
      check(A);
      bug(A);
    }
  )",
      "input.cpp"));
}
