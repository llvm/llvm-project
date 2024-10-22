//===- MemRegionDescriptiveNameTest.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "gtest/gtest.h"
#include <fstream>

using namespace clang;
using namespace ento;

namespace {

class DescriptiveNameChecker : public Checker<check::PreCall> {
public:
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const {
    if (!HandlerFn.matches(Call))
      return;

    const MemRegion *ArgReg = Call.getArgSVal(0).getAsRegion();
    assert(ArgReg && "expecting a location as the first argument");

    auto DescriptiveName = ArgReg->getDescriptiveName(/*UseQuotes=*/false);
    if (ExplodedNode *Node = C.generateNonFatalErrorNode(C.getState())) {
      auto Report =
          std::make_unique<PathSensitiveBugReport>(Bug, DescriptiveName, Node);
      C.emitReport(std::move(Report));
    }
  }

private:
  const BugType Bug{this, "DescriptiveNameBug"};
  const CallDescription HandlerFn = {
      CDM::SimpleFunc, {"reportDescriptiveName"}, 1};
};

void addDescriptiveNameChecker(AnalysisASTConsumer &AnalysisConsumer,
                               AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"DescriptiveNameChecker", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker<DescriptiveNameChecker>("DescriptiveNameChecker",
                                                "Desc", "DocsURI");
  });
}

bool runChecker(StringRef Code, std::string &Output) {
  return runCheckerOnCode<addDescriptiveNameChecker>(Code.str(), Output,
                                                     /*OnlyEmitWarnings=*/true);
}

TEST(MemRegionDescriptiveNameTest, ConcreteIntElementRegionIndex) {
  StringRef Code = R"cpp(
void reportDescriptiveName(int *p);
const unsigned int index = 1;
extern int array[3];
void top() {
  reportDescriptiveName(&array[index]);
})cpp";

  std::string Output;
  ASSERT_TRUE(runChecker(Code, Output));
  EXPECT_EQ(Output, "DescriptiveNameChecker: array[1]\n");
}

TEST(MemRegionDescriptiveNameTest, SymbolicElementRegionIndex) {
  StringRef Code = R"cpp(
void reportDescriptiveName(int *p);
extern unsigned int index;
extern int array[3];
void top() {
  reportDescriptiveName(&array[index]);
})cpp";

  std::string Output;
  ASSERT_TRUE(runChecker(Code, Output));
  EXPECT_EQ(Output, "DescriptiveNameChecker: array[index]\n");
}

TEST(MemRegionDescriptiveNameTest, SymbolicElementRegionIndexSymbolValFails) {
  StringRef Code = R"cpp(
void reportDescriptiveName(int *p);
extern int* ptr;
extern int array[3];
void top() {
  reportDescriptiveName(&array[(long long)ptr]);
})cpp";

  std::string Output;
  ASSERT_TRUE(runChecker(Code, Output));
  EXPECT_EQ(Output, "DescriptiveNameChecker: \n");
}

TEST(MemRegionDescriptiveNameTest, SymbolicElementRegionIndexOrigRegionFails) {
  StringRef Code = R"cpp(
void reportDescriptiveName(int *p);
extern int getInt(void);
extern int array[3];
void top() {
  reportDescriptiveName(&array[getInt()]);
})cpp";

  std::string Output;
  ASSERT_TRUE(runChecker(Code, Output));
  EXPECT_EQ(Output, "DescriptiveNameChecker: \n");
}

TEST(MemRegionDescriptiveNameTest, SymbolicElementRegionIndexDescrNameFails) {
  StringRef Code = R"cpp(
void reportDescriptiveName(int *p);
extern int *ptr;
extern int array[3];
void top() {
  reportDescriptiveName(&array[*ptr]);
})cpp";

  std::string Output;
  ASSERT_TRUE(runChecker(Code, Output));
  EXPECT_EQ(Output, "DescriptiveNameChecker: \n");
}

TEST(MemRegionDescriptiveNameTest,
     SymbolicElementRegionIndexIncorrectSymbolName) {
  StringRef Code = R"cpp(
void reportDescriptiveName(int *p);
extern int x, y;
extern int array[3];
void top() {
  y = x;
  reportDescriptiveName(&array[y]);
})cpp";

  std::string Output;
  ASSERT_TRUE(runChecker(Code, Output));
  // FIXME: Should return array[y], but returns array[x] (OriginRegion).
  EXPECT_EQ(Output, "DescriptiveNameChecker: array[x]\n");
}

} // namespace
