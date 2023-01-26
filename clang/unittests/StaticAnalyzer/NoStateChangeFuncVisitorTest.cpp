//===- unittests/StaticAnalyzer/NoStateChangeFuncVisitorTest.cpp ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CheckerRegistration.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporterVisitors.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/BugReporter/CommonBugCategories.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExplodedGraph.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <memory>

//===----------------------------------------------------------------------===//
// Base classes for testing NoStateChangeFuncVisitor.
//
// Testing is done by observing a very simple trait change from one node to
// another -- the checker sets the ErrorPrevented trait to true if
// 'preventError()' is called in the source code, and sets it to false if
// 'allowError()' is called. If this trait is false when 'error()' is called,
// a warning is emitted.
//
// The checker then registers a simple NoStateChangeFuncVisitor to add notes to
// inlined functions that could have, but neglected to prevent the error.
//===----------------------------------------------------------------------===//

REGISTER_TRAIT_WITH_PROGRAMSTATE(ErrorPrevented, bool)

namespace clang {
namespace ento {
namespace {

class ErrorNotPreventedFuncVisitor : public NoStateChangeFuncVisitor {
public:
  ErrorNotPreventedFuncVisitor()
      : NoStateChangeFuncVisitor(bugreporter::TrackingKind::Thorough) {}

  virtual PathDiagnosticPieceRef
  maybeEmitNoteForObjCSelf(PathSensitiveBugReport &R,
                           const ObjCMethodCall &Call,
                           const ExplodedNode *N) override {
    return nullptr;
  }

  virtual PathDiagnosticPieceRef
  maybeEmitNoteForCXXThis(PathSensitiveBugReport &R,
                          const CXXConstructorCall &Call,
                          const ExplodedNode *N) override {
    return nullptr;
  }

  virtual PathDiagnosticPieceRef
  maybeEmitNoteForParameters(PathSensitiveBugReport &R, const CallEvent &Call,
                             const ExplodedNode *N) override {
    PathDiagnosticLocation L = PathDiagnosticLocation::create(
        N->getLocation(),
        N->getState()->getStateManager().getContext().getSourceManager());
    return std::make_shared<PathDiagnosticEventPiece>(
        L, "Returning without prevening the error");
  }

  void Profile(llvm::FoldingSetNodeID &ID) const override {
    static int Tag = 0;
    ID.AddPointer(&Tag);
  }
};

template <class Visitor>
class StatefulChecker : public Checker<check::PreCall> {
  mutable std::unique_ptr<BugType> BT;

public:
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const {
    if (CallDescription{{"preventError"}, 0}.matches(Call)) {
      C.addTransition(C.getState()->set<ErrorPrevented>(true));
      return;
    }

    if (CallDescription{{"allowError"}, 0}.matches(Call)) {
      C.addTransition(C.getState()->set<ErrorPrevented>(false));
      return;
    }

    if (CallDescription{{"error"}, 0}.matches(Call)) {
      if (C.getState()->get<ErrorPrevented>())
        return;
      const ExplodedNode *N = C.generateErrorNode();
      if (!N)
        return;
      if (!BT)
        BT.reset(new BugType(this->getCheckerName(), "error()",
                             categories::SecurityError));
      auto R =
          std::make_unique<PathSensitiveBugReport>(*BT, "error() called", N);
      R->template addVisitor<Visitor>();
      C.emitReport(std::move(R));
    }
  }
};

} // namespace
} // namespace ento
} // namespace clang

//===----------------------------------------------------------------------===//
// Non-thorough analysis: only the state right before and right after the
// function call is checked for the difference in trait value.
//===----------------------------------------------------------------------===//

namespace clang {
namespace ento {
namespace {

class NonThoroughErrorNotPreventedFuncVisitor
    : public ErrorNotPreventedFuncVisitor {
public:
  virtual bool
  wasModifiedInFunction(const ExplodedNode *CallEnterN,
                        const ExplodedNode *CallExitEndN) override {
    return CallEnterN->getState()->get<ErrorPrevented>() !=
           CallExitEndN->getState()->get<ErrorPrevented>();
  }
};

void addNonThoroughStatefulChecker(AnalysisASTConsumer &AnalysisConsumer,
                                   AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"test.StatefulChecker", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry
        .addChecker<StatefulChecker<NonThoroughErrorNotPreventedFuncVisitor>>(
            "test.StatefulChecker", "Description", "");
  });
}

TEST(NoStateChangeFuncVisitor, NonThoroughFunctionAnalysis) {
  std::string Diags;
  EXPECT_TRUE(runCheckerOnCode<addNonThoroughStatefulChecker>(R"(
    void error();
    void preventError();
    void allowError();

    void g() {
      //preventError();
    }

    void f() {
      g();
      error();
    }
  )", Diags));
  EXPECT_EQ(Diags,
            "test.StatefulChecker: Calling 'g' | Returning without prevening "
            "the error | Returning from 'g' | error() called\n");

  Diags.clear();

  EXPECT_TRUE(runCheckerOnCode<addNonThoroughStatefulChecker>(R"(
    void error();
    void preventError();
    void allowError();

    void g() {
      preventError();
      allowError();
    }

    void f() {
      g();
      error();
    }
  )", Diags));
  EXPECT_EQ(Diags,
            "test.StatefulChecker: Calling 'g' | Returning without prevening "
            "the error | Returning from 'g' | error() called\n");

  Diags.clear();

  EXPECT_TRUE(runCheckerOnCode<addNonThoroughStatefulChecker>(R"(
    void error();
    void preventError();
    void allowError();

    void g() {
      preventError();
    }

    void f() {
      g();
      error();
    }
  )", Diags));
  EXPECT_EQ(Diags, "");
}

} // namespace
} // namespace ento
} // namespace clang

//===----------------------------------------------------------------------===//
// Thorough analysis: only the state right before and right after the
// function call is checked for the difference in trait value.
//===----------------------------------------------------------------------===//

namespace clang {
namespace ento {
namespace {

class ThoroughErrorNotPreventedFuncVisitor
    : public ErrorNotPreventedFuncVisitor {
public:
  virtual bool
  wasModifiedBeforeCallExit(const ExplodedNode *CurrN,
                            const ExplodedNode *CallExitBeginN) override {
    return CurrN->getState()->get<ErrorPrevented>() !=
           CallExitBeginN->getState()->get<ErrorPrevented>();
  }
};

void addThoroughStatefulChecker(AnalysisASTConsumer &AnalysisConsumer,
                                AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"test.StatefulChecker", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker<StatefulChecker<ThoroughErrorNotPreventedFuncVisitor>>(
        "test.StatefulChecker", "Description", "");
  });
}

TEST(NoStateChangeFuncVisitor, ThoroughFunctionAnalysis) {
  std::string Diags;
  EXPECT_TRUE(runCheckerOnCode<addThoroughStatefulChecker>(R"(
    void error();
    void preventError();
    void allowError();

    void g() {
      //preventError();
    }

    void f() {
      g();
      error();
    }
  )", Diags));
  EXPECT_EQ(Diags,
            "test.StatefulChecker: Calling 'g' | Returning without prevening "
            "the error | Returning from 'g' | error() called\n");

  Diags.clear();

  EXPECT_TRUE(runCheckerOnCode<addThoroughStatefulChecker>(R"(
    void error();
    void preventError();
    void allowError();

    void g() {
      preventError();
      allowError();
    }

    void f() {
      g();
      error();
    }
  )", Diags));
  EXPECT_EQ(Diags, "test.StatefulChecker: error() called\n");

  Diags.clear();

  EXPECT_TRUE(runCheckerOnCode<addThoroughStatefulChecker>(R"(
    void error();
    void preventError();
    void allowError();

    void g() {
      preventError();
    }

    void f() {
      g();
      error();
    }
  )", Diags));
  EXPECT_EQ(Diags, "");
}

} // namespace
} // namespace ento
} // namespace clang
