//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ento;

// Some dummy trait that we can mutate back and forth to force a new State.
REGISTER_TRAIT_WITH_PROGRAMSTATE(Flag, bool)

namespace {
class FlipFlagOnCheckLocation : public Checker<check::Location> {
public:
  // We make sure we alter the State every time we model a checkLocation event.
  void checkLocation(SVal l, bool isLoad, const Stmt *S,
                     CheckerContext &C) const {
    ProgramStateRef State = C.getState();
    State = State->set<Flag>(!State->get<Flag>());
    C.addTransition(State);
  }
};

void addFlagFlipperChecker(AnalysisASTConsumer &AnalysisConsumer,
                           AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"test.FlipFlagOnCheckLocation", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker<FlipFlagOnCheckLocation>("test.FlipFlagOnCheckLocation",
                                                 "Description", "");
  });
}

TEST(ObjCTest, CheckLocationEventsShouldMaterializeInObjCForCollectionStmts) {
  // Previously, the `ExprEngine::hasMoreIteration` may fired an assertion
  // because we forgot to handle correctly the resulting nodes of the
  // check::Location callback for the ObjCForCollectionStmts.
  // This caused inconsistencies in the graph and triggering the assertion.
  // See #124477 for more details.
  std::string Diags;
  EXPECT_TRUE(runCheckerOnCodeWithArgs<addFlagFlipperChecker>(
      R"(
    @class NSArray, NSDictionary, NSString;
    extern void NSLog(NSString *format, ...) __attribute__((format(__NSString__, 1, 2)));
    void entrypoint(NSArray *bowl) {
      for (NSString *fruit in bowl) { // no-crash
        NSLog(@"Fruit: %@", fruit);
      }
    })",
      {"-x", "objective-c"}, Diags));
}

} // namespace
