//===-- Unittests for VDSO ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/time_macros.h"
#include "hdr/types/clockid_t.h"
#include "hdr/types/struct_timespec.h"
#include "hdr/types/struct_timeval.h"
#include "include/llvm-libc-macros/linux/time-macros.h"
#include "src/__support/OSUtil/linux/vdso.h"
#include "test/UnitTest/LibcTest.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE {
TEST(LlvmLibcOSUtilVDSOTest, SymbolsDefined) {
  for (size_t i = 0; i < static_cast<size_t>(vdso::VDSOSym::VDSOSymCount);
       ++i) {
    // riscv_hwprobe is provided only on >=6.4 kernels. Skip it for now.
#ifdef LIBC_VDSO_HAS_RISCV_HWPROBE
    if (static_cast<vdso::VDSOSym>(i) == vdso::VDSOSym::RiscvHwProbe)
      continue;
#endif
    EXPECT_NE(vdso::get_symbol(static_cast<vdso::VDSOSym>(i)),
              static_cast<void *>(nullptr));
  }
}

#ifdef LIBC_VDSO_HAS_GETTIMEOFDAY
TEST(LlvmLibcOSUtilVDSOTest, GetTimeOfDay) {
  using FuncTy = int (*)(timeval *, struct timezone *);
  auto func =
      reinterpret_cast<FuncTy>(vdso::get_symbol(vdso::VDSOSym::GetTimeOfDay));
  timeval tv;
  EXPECT_EQ(func(&tv, nullptr), 0);
  // hopefully people are not building time machines using our libc.
  EXPECT_GT(tv.tv_sec, static_cast<decltype(tv.tv_sec)>(0));
}
#endif

#ifdef LIBC_VDSO_HAS_CLOCK_GETTIME
TEST(LlvmLibcOSUtilVDSOTest, ClockGetTime) {
  using FuncTy = int (*)(clockid_t, timespec *);
  auto func =
      reinterpret_cast<FuncTy>(vdso::get_symbol(vdso::VDSOSym::ClockGetTime));
  timespec a, b;
  EXPECT_EQ(func(CLOCK_MONOTONIC, &a), 0);
  EXPECT_EQ(func(CLOCK_MONOTONIC, &b), 0);
  if (a.tv_sec == b.tv_sec) {
    EXPECT_LT(a.tv_nsec, b.tv_nsec);
  } else {
    EXPECT_LT(a.tv_sec, b.tv_sec);
  }
}
#endif

#ifdef LIBC_VDSO_HAS_CLOCK_GETRES
TEST(LlvmLibcOSUtilVDSOTest, ClockGetRes) {
  using FuncTy = int (*)(clockid_t, timespec *);
  auto func =
      reinterpret_cast<FuncTy>(vdso::get_symbol(vdso::VDSOSym::ClockGetRes));
  timespec res{};
  EXPECT_EQ(func(CLOCK_MONOTONIC, &res), 0);
  EXPECT_TRUE(res.tv_sec > 0 || res.tv_nsec > 0);
}
#endif

} // namespace LIBC_NAMESPACE
