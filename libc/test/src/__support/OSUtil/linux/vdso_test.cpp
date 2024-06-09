//===-- Unittests for VDSO ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/signal_macros.h"
#include "hdr/time_macros.h"
#include "hdr/types/clockid_t.h"
#include "hdr/types/struct_sigaction.h"
#include "hdr/types/struct_timespec.h"
#include "hdr/types/struct_timeval.h"
#include "hdr/types/time_t.h"
#include "src/__support/OSUtil/linux/vdso.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/signal/raise.h"
#include "src/signal/sigaction.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/LibcTest.h" vdso_test
#include "test/UnitTest/Test.h"
#include <linux/time_types.h>
#include <sys/syscall.h>

namespace LIBC_NAMESPACE {
// For x86_64, we explicitly test some traditional vdso symbols are indeed
// available.

TEST(LlvmLibcOSUtilVDSOTest, GetTimeOfDay) {
  using FuncTy = int (*)(timeval *, struct timezone *);
  vdso::Symbol symbol{vdso::VDSOSym::GetTimeOfDay};
  auto func = symbol.get<FuncTy>();
#ifdef LIBC_TARGET_ARCH_IS_X86
  ASSERT_NE(func, static_cast<FuncTy>(nullptr));
#else
  if (func == nullptr)
    return;
#endif
  timeval tv;
  EXPECT_EQ(func(&tv, nullptr), 0);
  // hopefully people are not building time machines using our libc.
  EXPECT_GT(tv.tv_sec, static_cast<decltype(tv.tv_sec)>(0));
}

TEST(LlvmLibcOSUtilVDSOTest, Time) {
  using FuncTy = time_t (*)(time_t *);
  vdso::Symbol symbol{vdso::VDSOSym::Time};
  auto func = symbol.get<FuncTy>();
#ifdef LIBC_TARGET_ARCH_IS_X86
  ASSERT_NE(func, static_cast<FuncTy>(nullptr));
#else
  if (func == nullptr)
    return;
#endif
  time_t a, b;
  EXPECT_GT(func(&a), static_cast<time_t>(0));
  EXPECT_GT(func(&b), static_cast<time_t>(0));
  EXPECT_GE(b, a);
}

TEST(LlvmLibcOSUtilVDSOTest, ClockGetTime) {
  using FuncTy = int (*)(clockid_t, timespec *);
  vdso::Symbol symbol{vdso::VDSOSym::ClockGetTime};
  auto func = symbol.get<FuncTy>();
#ifdef LIBC_TARGET_ARCH_IS_X86
  ASSERT_NE(func, static_cast<FuncTy>(nullptr));
#else
  if (func == nullptr)
    return;
#endif
  timespec a, b;
  EXPECT_EQ(func(CLOCK_MONOTONIC, &a), 0);
  EXPECT_EQ(func(CLOCK_MONOTONIC, &b), 0);
  if (a.tv_sec == b.tv_sec) {
    EXPECT_LT(a.tv_nsec, b.tv_nsec);
  } else {
    EXPECT_LT(a.tv_sec, b.tv_sec);
  }
}

TEST(LlvmLibcOSUtilVDSOTest, ClockGetTime64) {
  using FuncTy = int (*)(clockid_t, __kernel_timespec *);
  vdso::Symbol symbol{vdso::VDSOSym::ClockGetTime64};
  auto func = symbol.get<FuncTy>();
  if (func == nullptr)
    return;
  // See kernel API at
  // https://elixir.bootlin.com/linux/latest/source/tools/testing/selftests/vDSO/vdso_test_correctness.c#L155
  __kernel_timespec a, b;
  EXPECT_EQ(func(CLOCK_MONOTONIC, &a), 0);
  EXPECT_EQ(func(CLOCK_MONOTONIC, &b), 0);
  if (a.tv_sec == b.tv_sec) {
    EXPECT_LT(a.tv_nsec, b.tv_nsec);
  } else {
    EXPECT_LT(a.tv_sec, b.tv_sec);
  }
}

TEST(LlvmLibcOSUtilVDSOTest, ClockGetRes) {
  using FuncTy = int (*)(clockid_t, timespec *);
  vdso::Symbol symbol{vdso::VDSOSym::ClockGetRes};
  auto func = symbol.get<FuncTy>();
  if (func == nullptr)
    return;
  timespec res{};
  EXPECT_EQ(func(CLOCK_MONOTONIC, &res), 0);
  EXPECT_TRUE(res.tv_sec > 0 || res.tv_nsec > 0);
}

TEST(LlvmLibcOSUtilVDSOTest, GetCpu) {
  // The kernel system call has a third argument, which should be passed as
  // nullptr.
  using FuncTy = int (*)(int *, int *, void *);
  vdso::Symbol symbol{vdso::VDSOSym::GetCpu};
  auto func = symbol.get<FuncTy>();
  if (func == nullptr)
    return;
  int cpu = -1, node = -1;
  EXPECT_EQ(func(&cpu, &node, nullptr), 0);
  EXPECT_GE(cpu, 0);
  EXPECT_GE(node, 0);
}

static bool flag = false;
static void sigprof_handler [[gnu::used]] (int) { flag = true; }

TEST(LlvmLibcOSUtilVDSOTest, RtSigReturn) {
  using namespace testing::ErrnoSetterMatcher;
  // must use struct since there is a function of the same name in the same
  // scope.
  struct sigaction sa {};
  struct sigaction old_sa {};
  sa.sa_handler = sigprof_handler;
  sa.sa_flags = SA_RESTORER;
  vdso::Symbol symbol{vdso::VDSOSym::RTSigReturn};
  auto func = symbol.get<decltype(sa.sa_restorer)>();
  if (func == nullptr)
    return;
  sa.sa_restorer = func;
  ASSERT_THAT(LIBC_NAMESPACE::sigaction(SIGPROF, &sa, &old_sa), Succeeds());
  raise(SIGPROF);
  ASSERT_TRUE(flag);
  flag = false;
  ASSERT_THAT(LIBC_NAMESPACE::sigaction(SIGPROF, &old_sa, nullptr), Succeeds());
}

TEST(LlvmLibcOSUtilVDSOTest, FlushICache) {
  using FuncTy = void (*)(void *, void *, unsigned long);
  vdso::Symbol symbol{vdso::VDSOSym::FlushICache};
  auto func = symbol.get<FuncTy>();
  if (func == nullptr)
    return;
  char buf[512];
  // we just check that the flush will not panic the program.
  // the flags part only take 0/1 as up to kernel 6.10, which is used to
  // indicate whether the flush is local to the core or global.
  func(buf, buf + sizeof(buf), 0);
  func(buf, buf + sizeof(buf), 1);
}

// https://docs.kernel.org/6.5/riscv/hwprobe.html
TEST(LlvmLibcOSUtilVDSOTest, RiscvHwProbe) {
  using namespace testing::ErrnoSetterMatcher;
  struct riscv_hwprobe {
    int64_t key;
    uint64_t value;
  };
  using FuncTy =
      long (*)(riscv_hwprobe *, size_t, size_t, struct cpu_set_t *, unsigned);
  vdso::Symbol symbol{vdso::VDSOSym::RiscvHwProbe};
  auto func = symbol.get<FuncTy>();
  if (func == nullptr)
    return;
  // If a key is unknown to the kernel, its key field will be cleared to -1, and
  // its value set to 0. We expect probes.value are all 0.
  // Usermode can supply NULL for cpus and 0 for cpu_count as a shortcut for all
  // online CPUs
  riscv_hwprobe probes[2] = {{-1, 1}, {-1, 1}};
  ASSERT_THAT(func(/*pairs=*/probes, /*count=*/2, /*cpusetsize=*/0,
                   /*cpuset=*/nullptr,
                   /*flags=*/0),
              Succeeds());
  for (auto &probe : probes) {
    EXPECT_EQ(probe.key, static_cast<decltype(probe.key)>(-1));
    EXPECT_EQ(probe.value, static_cast<decltype(probe.value)>(0));
  }
}

} // namespace LIBC_NAMESPACE
