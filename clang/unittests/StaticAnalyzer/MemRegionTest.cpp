//===- unittests/StaticAnalyzer/MemRegionTest.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CheckerRegistration.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

#include "Reusables.h"

namespace clang {
namespace ento {
namespace {

std::string MemRegName;

class MemRegChecker : public Checker<check::Location> {
public:
  void checkLocation(const SVal &Loc, bool IsLoad, const Stmt *S,
                     CheckerContext &CC) const {
    if (const MemRegion *MemReg = Loc.getAsRegion())
      MemRegName = MemReg->getDescriptiveName(false);
  }
};

void addMemRegChecker(AnalysisASTConsumer &AnalysisConsumer,
                      AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"test.MemRegChecker", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker<MemRegChecker>("test.MemRegChecker", "Description", "");
  });
}

TEST(MemRegion, DescriptiveName) {
  EXPECT_TRUE(runCheckerOnCode<addMemRegChecker>(
      "const unsigned int index = 1; "
      "extern int array[3]; "
      "int main() { int a = array[index]; return 0; }"));
  EXPECT_EQ(MemRegName, "array[1]");
  MemRegName.clear();

  EXPECT_TRUE(runCheckerOnCode<addMemRegChecker>(
      "extern unsigned int index; "
      "extern int array[3]; "
      "int main() { int a = array[index]; return 0; }"));
  EXPECT_EQ(MemRegName, "array[index]");
  MemRegName.clear();
}

} // namespace
} // namespace ento
} // namespace clang
