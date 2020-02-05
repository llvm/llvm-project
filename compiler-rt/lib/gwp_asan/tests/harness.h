//===-- harness.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GWP_ASAN_TESTS_HARNESS_H_
#define GWP_ASAN_TESTS_HARNESS_H_

#include <stdarg.h>

#include "gtest/gtest.h"

#include "gwp_asan/guarded_pool_allocator.h"
#include "gwp_asan/optional/backtrace.h"
#include "gwp_asan/options.h"

namespace gwp_asan {
namespace test {
// This printf-function getter allows other platforms (e.g. Android) to define
// their own signal-safe Printf function. In LLVM, we use
// `optional/printf_sanitizer_common.cpp` which supplies the __sanitizer::Printf
// for this purpose.
options::Printf_t getPrintfFunction();

// First call returns true, all the following calls return false.
bool OnlyOnce();

}; // namespace test
}; // namespace gwp_asan

class DefaultGuardedPoolAllocator : public ::testing::Test {
public:
  void SetUp() override {
    gwp_asan::options::Options Opts;
    Opts.setDefaults();
    MaxSimultaneousAllocations = Opts.MaxSimultaneousAllocations;

    Opts.Printf = gwp_asan::test::getPrintfFunction();
    Opts.InstallForkHandlers = gwp_asan::test::OnlyOnce();
    GPA.init(Opts);
  }

  void TearDown() override { GPA.uninitTestOnly(); }

protected:
  gwp_asan::GuardedPoolAllocator GPA;
  decltype(gwp_asan::options::Options::MaxSimultaneousAllocations)
      MaxSimultaneousAllocations;
};

class CustomGuardedPoolAllocator : public ::testing::Test {
public:
  void
  InitNumSlots(decltype(gwp_asan::options::Options::MaxSimultaneousAllocations)
                   MaxSimultaneousAllocationsArg) {
    gwp_asan::options::Options Opts;
    Opts.setDefaults();

    Opts.MaxSimultaneousAllocations = MaxSimultaneousAllocationsArg;
    MaxSimultaneousAllocations = MaxSimultaneousAllocationsArg;

    Opts.Printf = gwp_asan::test::getPrintfFunction();
    Opts.InstallForkHandlers = gwp_asan::test::OnlyOnce();
    GPA.init(Opts);
  }

  void TearDown() override { GPA.uninitTestOnly(); }

protected:
  gwp_asan::GuardedPoolAllocator GPA;
  decltype(gwp_asan::options::Options::MaxSimultaneousAllocations)
      MaxSimultaneousAllocations;
};

class BacktraceGuardedPoolAllocator : public ::testing::Test {
public:
  void SetUp() override {
    gwp_asan::options::Options Opts;
    Opts.setDefaults();

    Opts.Printf = gwp_asan::test::getPrintfFunction();
    Opts.Backtrace = gwp_asan::options::getBacktraceFunction();
    Opts.PrintBacktrace = gwp_asan::options::getPrintBacktraceFunction();
    Opts.InstallForkHandlers = gwp_asan::test::OnlyOnce();
    GPA.init(Opts);
  }

  void TearDown() override { GPA.uninitTestOnly(); }

protected:
  gwp_asan::GuardedPoolAllocator GPA;
};

#endif // GWP_ASAN_TESTS_HARNESS_H_
