//===-- harness.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GWP_ASAN_TESTS_HARNESS_H_
#define GWP_ASAN_TESTS_HARNESS_H_

#include "gtest/gtest.h"

// Include sanitizer_common first as gwp_asan/guarded_pool_allocator.h
// transiently includes definitions.h, which overwrites some of the definitions
// in sanitizer_common.
#include "sanitizer_common/sanitizer_common.h"

#include "gwp_asan/guarded_pool_allocator.h"
#include "gwp_asan/optional/backtrace.h"
#include "gwp_asan/optional/options_parser.h"
#include "gwp_asan/options.h"

class DefaultGuardedPoolAllocator : public ::testing::Test {
public:
  DefaultGuardedPoolAllocator() {
    gwp_asan::options::Options Opts;
    Opts.setDefaults();
    MaxSimultaneousAllocations = Opts.MaxSimultaneousAllocations;

    Opts.Printf = __sanitizer::Printf;
    GPA.init(Opts);
  }

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

    Opts.Printf = __sanitizer::Printf;
    GPA.init(Opts);
  }

protected:
  gwp_asan::GuardedPoolAllocator GPA;
  decltype(gwp_asan::options::Options::MaxSimultaneousAllocations)
      MaxSimultaneousAllocations;
};

class BacktraceGuardedPoolAllocator : public ::testing::Test {
public:
  BacktraceGuardedPoolAllocator() {
    // Call initOptions to initialise the internal sanitizer_common flags. These
    // flags are referenced by the sanitizer_common unwinder, and if left
    // uninitialised, they'll unintentionally crash the program.
    gwp_asan::options::initOptions();

    gwp_asan::options::Options Opts;
    Opts.setDefaults();

    Opts.Printf = __sanitizer::Printf;
    Opts.Backtrace = gwp_asan::options::getBacktraceFunction();
    Opts.PrintBacktrace = gwp_asan::options::getPrintBacktraceFunction();
    GPA.init(Opts);
  }

protected:
  gwp_asan::GuardedPoolAllocator GPA;
};

#endif // GWP_ASAN_TESTS_HARNESS_H_
