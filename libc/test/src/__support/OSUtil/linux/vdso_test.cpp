//===-- Unittests for VDSO ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_timeval.h"
#include "src/__support/OSUtil/vdso.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE {
TEST(LlvmLibcOSUtilVDSOTest, SymbolsDefined) {
  // for now, we simply test all symbols are provided.
  for (size_t i = 0; i < static_cast<size_t>(vdso::VDSOSym::VDSOSymCount); ++i)
    EXPECT_NE(vdso::get_symbol(static_cast<vdso::VDSOSym>(i)),
              static_cast<void *>(nullptr));
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

} // namespace LIBC_NAMESPACE
