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

#include <errno.h>
#include <string.h>
#include <sys/mman.h>

#include <algorithm>
#include <vector>

namespace scudo {

TEST(ScudoCommonTest, VerifyGetResidentPages) {
  if (!SCUDO_LINUX)
    GTEST_SKIP() << "Only valid on linux systems.";

  constexpr uptr NumPages = 512;
  const uptr SizeBytes = NumPages * getPageSizeCached();

  MemMapT MemMap;
  ASSERT_TRUE(MemMap.map(/*Addr=*/0U, SizeBytes, "ResidentMemorySize"));
  ASSERT_NE(MemMap.getBase(), 0U);

  // Only android seems to properly detect when single pages are touched.
#if SCUDO_ANDROID
  // Verify nothing should be mapped in right after the map is created.
  EXPECT_EQ(0U, getResidentPages(MemMap.getBase(), SizeBytes));

  // Touch a page.
  u8 *Data = reinterpret_cast<u8 *>(MemMap.getBase());
  Data[0] = 1;
  EXPECT_EQ(1U, getResidentPages(MemMap.getBase(), SizeBytes));

  // Touch a non-consective page.
  Data[getPageSizeCached() * 2] = 1;
  EXPECT_EQ(2U, getResidentPages(MemMap.getBase(), SizeBytes));

  // Touch a page far enough that the function has to make multiple calls
  // to mincore.
  Data[getPageSizeCached() * 300] = 1;
  EXPECT_EQ(3U, getResidentPages(MemMap.getBase(), SizeBytes));

  // Touch another page in the same range to make sure the second
  // read is working.
  Data[getPageSizeCached() * 400] = 1;
  EXPECT_EQ(4U, getResidentPages(MemMap.getBase(), SizeBytes));
#endif

  // Now write the whole thing.
  memset(reinterpret_cast<void *>(MemMap.getBase()), 1, SizeBytes);
  EXPECT_EQ(NumPages, getResidentPages(MemMap.getBase(), SizeBytes));

  MemMap.unmap();
}

TEST(ScudoCommonTest, VerifyReleasePagesToOS) {
  if (!SCUDO_LINUX)
    GTEST_SKIP() << "Only valid on linux systems.";

  constexpr uptr NumPages = 1000;
  const uptr SizeBytes = NumPages * getPageSizeCached();

  MemMapT MemMap;
  ASSERT_TRUE(MemMap.map(/*Addr=*/0U, SizeBytes, "ResidentMemorySize"));
  ASSERT_NE(MemMap.getBase(), 0U);

  void *P = reinterpret_cast<void *>(MemMap.getBase());
  EXPECT_EQ(0U, getResidentPages(MemMap.getBase(), SizeBytes));

  // Make the entire map resident.
  memset(P, 1, SizeBytes);
  EXPECT_EQ(NumPages, getResidentPages(MemMap.getBase(), SizeBytes));

  // Should release the memory to the kernel immediately.
  MemMap.releasePagesToOS(MemMap.getBase(), SizeBytes);
  EXPECT_EQ(0U, getResidentPages(MemMap.getBase(), SizeBytes));

  // Make the entire map resident again.
  memset(P, 1, SizeBytes);
  EXPECT_EQ(NumPages, getResidentPages(MemMap.getBase(), SizeBytes));

  MemMap.unmap();
}

TEST(ScudoCommonTest, Zeros) {
  const uptr Size = 1ull << 20;

  MemMapT MemMap;
  ASSERT_TRUE(MemMap.map(/*Addr=*/0U, Size, "Zeros"));
  ASSERT_NE(MemMap.getBase(), 0U);
  uptr *P = reinterpret_cast<uptr *>(MemMap.getBase());
  const ptrdiff_t N = Size / sizeof(uptr);
  EXPECT_EQ(std::count(P, P + N, 0), N);

  memset(P, 1, Size);
  EXPECT_EQ(std::count(P, P + N, 0), 0);

  MemMap.releasePagesToOS(MemMap.getBase(), Size);
  EXPECT_EQ(std::count(P, P + N, 0), N);

  MemMap.unmap();
}

} // namespace scudo
