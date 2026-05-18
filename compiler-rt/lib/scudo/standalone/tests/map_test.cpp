//===-- map_test.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tests/scudo_unit_test.h"

#include "common.h"
#include "mem_map.h"

#include <algorithm>
#include <string.h>
#include <unistd.h>

#if SCUDO_LINUX
#include <sys/mman.h>
#endif

static const char *MappingName = "scudo:test";

TEST(ScudoMapTest, PageSize) {
  EXPECT_EQ(scudo::getPageSizeCached(),
            static_cast<scudo::uptr>(sysconf(_SC_PAGESIZE)));
}

TEST(ScudoMapTest, VerifyGetResidentPages) {
  if (!SCUDO_LINUX)
    TEST_SKIP("Only valid on linux systems.");

  constexpr scudo::uptr NumPages = 512;
  const scudo::uptr SizeBytes = NumPages * scudo::getPageSizeCached();

  scudo::MemMapT MemMap;
  ASSERT_TRUE(MemMap.map(/*Addr=*/0U, SizeBytes, "ResidentMemorySize"));
  ASSERT_NE(MemMap.getBase(), 0U);

  // Only android seems to properly detect when single pages are touched.
#if SCUDO_ANDROID
  // Verify nothing should be mapped in right after the map is created.
  EXPECT_EQ(0U, MemMap.getResidentPages(MemMap.getBase(), SizeBytes));

  // Touch a page.
  scudo::u8 *Data = reinterpret_cast<scudo::u8 *>(MemMap.getBase());
  Data[0] = 1;
  EXPECT_EQ(1U, MemMap.getResidentPages(MemMap.getBase(), SizeBytes));

  // Touch a non-consective page.
  Data[scudo::getPageSizeCached() * 2] = 1;
  EXPECT_EQ(2U, MemMap.getResidentPages(MemMap.getBase(), SizeBytes));

  // Touch a page far enough that the function has to make multiple calls
  // to mincore.
  Data[scudo::getPageSizeCached() * 300] = 1;
  EXPECT_EQ(3U, MemMap.getResidentPages(MemMap.getBase(), SizeBytes));

  // Touch another page in the same range to make sure the second
  // read is working.
  Data[scudo::getPageSizeCached() * 400] = 1;
  EXPECT_EQ(4U, MemMap.getResidentPages(MemMap.getBase(), SizeBytes));
#endif

  // Now write the whole thing.
  memset(reinterpret_cast<void *>(MemMap.getBase()), 1, SizeBytes);
  scudo::s64 ResidentPages =
      MemMap.getResidentPages(MemMap.getBase(), SizeBytes);
  EXPECT_EQ(NumPages, static_cast<uintptr_t>(ResidentPages));

  MemMap.unmap();
}

TEST(ScudoMapTest, VerifyReleasePagesToOS) {
  if (!SCUDO_LINUX)
    TEST_SKIP("Only valid on linux systems.");

  constexpr scudo::uptr NumPages = 1000;
  const scudo::uptr SizeBytes = NumPages * scudo::getPageSizeCached();

  scudo::MemMapT MemMap;
  ASSERT_TRUE(MemMap.map(/*Addr=*/0U, SizeBytes, "ResidentMemorySize"));
  ASSERT_NE(MemMap.getBase(), 0U);

  void *P = reinterpret_cast<void *>(MemMap.getBase());
  EXPECT_EQ(0U, MemMap.getResidentPages(MemMap.getBase(), SizeBytes));

  // Make the entire map resident.
  memset(P, 1, SizeBytes);
  scudo::s64 ResidentPages =
      MemMap.getResidentPages(MemMap.getBase(), SizeBytes);
  if (ResidentPages >= 0)
    EXPECT_EQ(NumPages, static_cast<uintptr_t>(ResidentPages));

  // Should release the memory to the kernel immediately.
  MemMap.releasePagesToOS(MemMap.getBase(), SizeBytes);
  EXPECT_EQ(0U, MemMap.getResidentPages(MemMap.getBase(), SizeBytes));

  // Make the entire map resident again.
  memset(P, 1, SizeBytes);
  ResidentPages = MemMap.getResidentPages(MemMap.getBase(), SizeBytes);
  EXPECT_EQ(NumPages, static_cast<uintptr_t>(ResidentPages));

  MemMap.unmap();
}

TEST(ScudoMapTest, Zeros) {
  const scudo::uptr Size = 1ull << 20;

  scudo::MemMapT MemMap;
  ASSERT_TRUE(MemMap.map(/*Addr=*/0U, Size, "Zeros"));
  ASSERT_NE(MemMap.getBase(), 0U);
  scudo::uptr *P = reinterpret_cast<scudo::uptr *>(MemMap.getBase());
  const ptrdiff_t N = Size / sizeof(scudo::uptr);
  EXPECT_EQ(std::count(P, P + N, 0), N);

  memset(P, 1, Size);
  EXPECT_EQ(std::count(P, P + N, 0), 0);

  MemMap.releasePagesToOS(MemMap.getBase(), Size);
  EXPECT_EQ(std::count(P, P + N, 0), N);

  MemMap.unmap();
}

TEST(ScudoMapDeathTest, MapNoAccessUnmap) {
  const scudo::uptr Size = 4 * scudo::getPageSizeCached();
  scudo::ReservedMemoryT ReservedMemory;

  ASSERT_TRUE(ReservedMemory.create(/*Addr=*/0U, Size, MappingName));
  EXPECT_NE(ReservedMemory.getBase(), 0U);
  EXPECT_DEATH(
      memset(reinterpret_cast<void *>(ReservedMemory.getBase()), 0xaa, Size),
      "");

  ReservedMemory.release();
}

TEST(ScudoMapDeathTest, MapUnmap) {
  const scudo::uptr Size = 4 * scudo::getPageSizeCached();
  EXPECT_DEATH(
      {
        // Repeat few time to avoid missing crash if it's mmaped by unrelated
        // code.
        for (int i = 0; i < 10; ++i) {
          scudo::MemMapT MemMap;
          MemMap.map(/*Addr=*/0U, Size, MappingName);
          scudo::uptr P = MemMap.getBase();
          if (P == 0U)
            continue;
          MemMap.unmap();
          memset(reinterpret_cast<void *>(P), 0xbb, Size);
        }
      },
      "");
}

TEST(ScudoMapDeathTest, MapWithGuardUnmap) {
  const scudo::uptr PageSize = scudo::getPageSizeCached();
  const scudo::uptr Size = 4 * PageSize;
  scudo::ReservedMemoryT ReservedMemory;
  ASSERT_TRUE(
      ReservedMemory.create(/*Addr=*/0U, Size + 2 * PageSize, MappingName));
  ASSERT_NE(ReservedMemory.getBase(), 0U);

  scudo::MemMapT MemMap =
      ReservedMemory.dispatch(ReservedMemory.getBase(), Size + 2 * PageSize);
  ASSERT_TRUE(MemMap.isAllocated());
  scudo::uptr Q = MemMap.getBase() + PageSize;
  ASSERT_TRUE(MemMap.remap(Q, Size, MappingName));
  memset(reinterpret_cast<void *>(Q), 0xaa, Size);
  EXPECT_DEATH(memset(reinterpret_cast<void *>(Q), 0xaa, Size + 1), "");
  MemMap.unmap();
}

// These death tests only fail when debugging is enabled.
#if SCUDO_LINUX
TEST(ScudoMapDeathTest, ResidentPagesNotMapped) {
  scudo::MemMapT MemMap;
  ASSERT_EQ(MemMap.getResidentPages(), 0);
}
#endif

TEST(ScudoMapTest, MapGrowUnmap) {
  const scudo::uptr PageSize = scudo::getPageSizeCached();
  const scudo::uptr Size = 4 * PageSize;
  scudo::ReservedMemoryT ReservedMemory;
  ReservedMemory.create(/*Addr=*/0U, Size, MappingName);
  ASSERT_TRUE(ReservedMemory.isCreated());

  scudo::MemMapT MemMap =
      ReservedMemory.dispatch(ReservedMemory.getBase(), Size);
  ASSERT_TRUE(MemMap.isAllocated());
  scudo::uptr Q = MemMap.getBase() + PageSize;
  ASSERT_TRUE(MemMap.remap(Q, PageSize, MappingName));
  memset(reinterpret_cast<void *>(Q), 0xaa, PageSize);
  Q += PageSize;
  ASSERT_TRUE(MemMap.remap(Q, PageSize, MappingName));
  memset(reinterpret_cast<void *>(Q), 0xbb, PageSize);
  MemMap.unmap();
}

// Verify that zeroing works properly.
TEST(ScudoMapTest, Zeroing) {
  scudo::ReservedMemoryT ReservedMemory;
  const scudo::uptr PageSize = scudo::getPageSizeCached();
  const scudo::uptr Size = 3 * PageSize;
  ReservedMemory.create(/*Addr=*/0U, Size, MappingName);
  ASSERT_TRUE(ReservedMemory.isCreated());

  scudo::MemMapT MemMap = ReservedMemory.dispatch(ReservedMemory.getBase(),
                                                  ReservedMemory.getCapacity());
  EXPECT_TRUE(
      MemMap.remap(MemMap.getBase(), MemMap.getCapacity(), MappingName));
  unsigned char *Data = reinterpret_cast<unsigned char *>(MemMap.getBase());
  memset(Data, 1U, MemMap.getCapacity());
  // Spot check some values.
  EXPECT_EQ(1U, Data[0]);
  EXPECT_EQ(1U, Data[PageSize]);
  EXPECT_EQ(1U, Data[PageSize * 2]);
  MemMap.releaseAndZeroPagesToOS(MemMap.getBase(), MemMap.getCapacity());
  EXPECT_EQ(0U, Data[0]);
  EXPECT_EQ(0U, Data[PageSize]);
  EXPECT_EQ(0U, Data[PageSize * 2]);

#if SCUDO_LINUX
  // Now verify that if madvise fails, the data is still zeroed.
  memset(Data, 1U, MemMap.getCapacity());
  if (mlock(Data, MemMap.getCapacity()) != -1) {
    EXPECT_EQ(1U, Data[0]);
    EXPECT_EQ(1U, Data[PageSize]);
    EXPECT_EQ(1U, Data[PageSize * 2]);
    MemMap.releaseAndZeroPagesToOS(MemMap.getBase(), MemMap.getCapacity());
    EXPECT_EQ(0U, Data[0]);
    EXPECT_EQ(0U, Data[PageSize]);
    EXPECT_EQ(0U, Data[PageSize * 2]);

    EXPECT_NE(-1, munlock(Data, MemMap.getCapacity()));
  }
#endif

  MemMap.unmap();
}
