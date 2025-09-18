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

static void getResidentPages(void *BaseAddress, size_t TotalPages,
                             size_t *ResidentPages) {
  std::vector<unsigned char> Pages(TotalPages, 0);
  ASSERT_EQ(
      0, mincore(BaseAddress, TotalPages * getPageSizeCached(), Pages.data()))
      << strerror(errno);
  *ResidentPages = 0;
  for (unsigned char Value : Pages) {
    if (Value & 1) {
      ++*ResidentPages;
    }
  }
}

// Fuchsia needs getResidentPages implementation.
TEST(ScudoCommonTest, SKIP_ON_FUCHSIA(ResidentMemorySize)) {
  // Make sure to have the size of the map on a page boundary.
  const uptr PageSize = getPageSizeCached();
  const size_t NumPages = 1000;
  const uptr SizeBytes = NumPages * PageSize;

  MemMapT MemMap;
  ASSERT_TRUE(MemMap.map(/*Addr=*/0U, SizeBytes, "ResidentMemorySize"));
  ASSERT_NE(MemMap.getBase(), 0U);

  void *P = reinterpret_cast<void *>(MemMap.getBase());
  size_t ResidentPages;
  getResidentPages(P, NumPages, &ResidentPages);
  EXPECT_EQ(0U, ResidentPages);

  // Make the entire map resident.
  memset(P, 1, SizeBytes);
  getResidentPages(P, NumPages, &ResidentPages);
  EXPECT_EQ(NumPages, ResidentPages);

  // Should release the memory to the kernel immediately.
  MemMap.releasePagesToOS(MemMap.getBase(), SizeBytes);
  getResidentPages(P, NumPages, &ResidentPages);
  EXPECT_EQ(0U, ResidentPages);

  // Make the entire map resident again.
  memset(P, 1, SizeBytes);
  getResidentPages(P, NumPages, &ResidentPages);
  EXPECT_EQ(NumPages, ResidentPages);

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
