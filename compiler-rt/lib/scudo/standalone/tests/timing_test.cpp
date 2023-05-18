//===-- timing_test.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tests/scudo_unit_test.h"

#include "timing.h"

#include <string>

class ScudoTimingTest : public Test {
public:
  void testFunc1() { scudo::ScopedTimer ST(Manager, __func__); }

  void testFunc2() {
    scudo::ScopedTimer ST(Manager, __func__);
    testFunc1();
  }

  void testChainedCalls() {
    scudo::ScopedTimer ST(Manager, __func__);
    testFunc2();
  }

  void testIgnoredTimer() {
    scudo::ScopedTimer ST(Manager, __func__);
    ST.ignore();
  }

  void printAllTimersStats() { Manager.printAll(); }

  scudo::TimingManager &getTimingManager() { return Manager; }

private:
  scudo::TimingManager Manager;
};

// Given that the output of statistics of timers are dumped through
// `scudo::Printf` which is platform dependent, so we don't have a reliable way
// to catch the output and verify the details. Now we only verify the number of
// invocations on linux.
TEST_F(ScudoTimingTest, SimpleTimer) {
#if SCUDO_LINUX
  testing::internal::LogToStderr();
  testing::internal::CaptureStderr();
#endif

  testIgnoredTimer();
  testChainedCalls();
  printAllTimersStats();

#if SCUDO_LINUX
  std::string output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(output.find("testIgnoredTimer (1)") == std::string::npos);
  EXPECT_TRUE(output.find("testChainedCalls (1)") != std::string::npos);
  EXPECT_TRUE(output.find("testFunc2 (1)") != std::string::npos);
  EXPECT_TRUE(output.find("testFunc1 (1)") != std::string::npos);
#endif
}

TEST_F(ScudoTimingTest, NestedTimer) {
#if SCUDO_LINUX
  testing::internal::LogToStderr();
  testing::internal::CaptureStderr();
#endif

  {
    scudo::ScopedTimer Outer(getTimingManager(), "Outer");
    {
      scudo::ScopedTimer Inner1(getTimingManager(), Outer, "Inner1");
      { scudo::ScopedTimer Inner2(getTimingManager(), Inner1, "Inner2"); }
    }
  }
  printAllTimersStats();

#if SCUDO_LINUX
  std::string output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(output.find("Outer (1)") != std::string::npos);
  EXPECT_TRUE(output.find("Inner1 (1)") != std::string::npos);
  EXPECT_TRUE(output.find("Inner2 (1)") != std::string::npos);
#endif
}
