//===-- common_test.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "internal_defs.h"
#include "tests/scudo_unit_test.h"

#include "common.h"
#include "mem_map.h"
#include <algorithm>
#include <fstream>

namespace scudo {

static uptr getResidentMemorySize() {
  if (!SCUDO_LINUX)
    UNREACHABLE("Not implemented!");
  uptr Size;
  uptr Resident;
  std::ifstream IFS("/proc/self/statm");
  IFS >> Size;
  IFS >> Resident;
  return Resident * getPageSizeCached();
}

// Fuchsia needs getResidentMemorySize implementation.
TEST(ScudoCommonTest, SKIP_ON_FUCHSIA(ResidentMemorySize)) {
  uptr OnStart = getResidentMemorySize();
  EXPECT_GT(OnStart, 0UL);

  const uptr Size = 1ull << 30;
  const uptr Threshold = Size >> 3;

  MemMapT MemMap;
  ASSERT_TRUE(MemMap.map(/*Addr=*/0U, Size, "ResidentMemorySize"));
  void *P = reinterpret_cast<void *>(MemMap.getBase());
  EXPECT_LT(getResidentMemorySize(), OnStart + Threshold);

  memset(P, 1, Size);
  EXPECT_GT(getResidentMemorySize(), OnStart + Size - Threshold);

  MemMap.releasePagesToOS(MemMap.getBase(), Size);
  EXPECT_LT(getResidentMemorySize(), OnStart + Threshold);

  memset(P, 1, Size);
  EXPECT_GT(getResidentMemorySize(), OnStart + Size - Threshold);

  MemMap.unmap(MemMap.getBase(), Size);
}

TEST(ScudoCommonTest, Zeros) {
  const uptr Size = 1ull << 20;

  MemMapT MemMap;
  ASSERT_TRUE(MemMap.map(/*Addr=*/0U, Size, "Zeros"));
  uptr *P = reinterpret_cast<uptr *>(MemMap.getBase());
  const ptrdiff_t N = Size / sizeof(uptr);
  EXPECT_EQ(std::count(P, P + N, 0), N);

  memset(P, 1, Size);
  EXPECT_EQ(std::count(P, P + N, 0), 0);

  MemMap.releasePagesToOS(MemMap.getBase(), Size);
  EXPECT_EQ(std::count(P, P + N, 0), N);

  MemMap.unmap(MemMap.getBase(), Size);
}

#if 0
// This test is temorarily disabled because it may not work as expected. E.g.,
// it doesn't dirty the pages so the pages may not be commited and it may only
// work on the single thread environment. As a result, this test is flaky and is
// impacting many test scenarios.
TEST(ScudoCommonTest, GetRssFromBuffer) {
  constexpr int64_t AllocSize = 10000000;
  constexpr int64_t Error = 3000000;
  constexpr size_t Runs = 10;

  int64_t Rss = scudo::GetRSS();
  EXPECT_GT(Rss, 0);

  std::vector<std::unique_ptr<char[]>> Allocs(Runs);
  for (auto &Alloc : Allocs) {
    Alloc.reset(new char[AllocSize]());
    int64_t Prev = Rss;
    Rss = scudo::GetRSS();
    EXPECT_LE(std::abs(Rss - AllocSize - Prev), Error);
  }
}
#endif

} // namespace scudo
