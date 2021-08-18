//===-- tsan_shadow_test.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#include "tsan_platform.h"
#include "tsan_rtl.h"
#include "gtest/gtest.h"

namespace __tsan {

TEST(Shadow, FastState) {
  Shadow s(FastState(11, 22));
  EXPECT_EQ(s.tid(), (u64)11);
  EXPECT_EQ(s.epoch(), (u64)22);
  EXPECT_EQ(s.GetIgnoreBit(), false);
  EXPECT_EQ(s.GetFreedAndReset(), false);
  EXPECT_EQ(s.GetHistorySize(), 0);
  EXPECT_EQ(s.addr0(), (u64)0);
  EXPECT_EQ(s.size(), (u64)1);
  EXPECT_EQ(s.IsWrite(), true);

  s.IncrementEpoch();
  EXPECT_EQ(s.epoch(), (u64)23);
  s.IncrementEpoch();
  EXPECT_EQ(s.epoch(), (u64)24);

  s.SetIgnoreBit();
  EXPECT_EQ(s.GetIgnoreBit(), true);
  s.ClearIgnoreBit();
  EXPECT_EQ(s.GetIgnoreBit(), false);

  for (int i = 0; i < 8; i++) {
    s.SetHistorySize(i);
    EXPECT_EQ(s.GetHistorySize(), i);
  }
  s.SetHistorySize(2);
  s.ClearHistorySize();
  EXPECT_EQ(s.GetHistorySize(), 0);
}

TEST(Shadow, Mapping) {
  static int global;
  int stack;
  void *heap = malloc(0);
  free(heap);

  CHECK(IsAppMem((uptr)&global));
  CHECK(IsAppMem((uptr)&stack));
  CHECK(IsAppMem((uptr)heap));

  CHECK(IsShadowMem(MemToShadow((uptr)&global)));
  CHECK(IsShadowMem(MemToShadow((uptr)&stack)));
  CHECK(IsShadowMem(MemToShadow((uptr)heap)));
}

TEST(Shadow, Celling) {
  u64 aligned_data[4];
  char *data = (char*)aligned_data;
  CHECK(IsAligned(reinterpret_cast<uptr>(data), kShadowSize));
  RawShadow *s0 = MemToShadow((uptr)&data[0]);
  CHECK(IsAligned(reinterpret_cast<uptr>(s0), kShadowSize));
  for (unsigned i = 1; i < kShadowCell; i++)
    CHECK_EQ(s0, MemToShadow((uptr)&data[i]));
  for (unsigned i = kShadowCell; i < 2*kShadowCell; i++)
    CHECK_EQ(s0 + kShadowCnt, MemToShadow((uptr)&data[i]));
  for (unsigned i = 2*kShadowCell; i < 3*kShadowCell; i++)
    CHECK_EQ(s0 + 2 * kShadowCnt, MemToShadow((uptr)&data[i]));
}

// Detect is the Mapping has kBroken field.
template <uptr>
struct Has {
  typedef bool Result;
};

template <typename Mapping>
bool broken(...) {
  return false;
}

template <typename Mapping>
bool broken(uptr what, typename Has<Mapping::kBroken>::Result = false) {
  return Mapping::kBroken & what;
}

struct MappingTest {
  template <typename Mapping>
  static void Apply() {
    // Easy (but ugly) way to print the mapping name.
    Printf("%s\n", __PRETTY_FUNCTION__);
    TestRegion<Mapping>(Mapping::kLoAppMemBeg, Mapping::kLoAppMemEnd);
    TestRegion<Mapping>(Mapping::kMidAppMemBeg, Mapping::kMidAppMemEnd);
    TestRegion<Mapping>(Mapping::kHiAppMemBeg, Mapping::kHiAppMemEnd);
    TestRegion<Mapping>(Mapping::kHeapMemBeg, Mapping::kHeapMemEnd);
  }

  template <typename Mapping>
  static void TestRegion(uptr beg, uptr end) {
    if (beg == end)
      return;
    Printf("checking region [0x%zx-0x%zx)\n", beg, end);
    uptr prev = 0;
    for (uptr p0 = beg; p0 <= end; p0 += (end - beg) / 256) {
      for (int x = -(int)kShadowCell; x <= (int)kShadowCell; x += kShadowCell) {
        const uptr p = RoundDown(p0 + x, kShadowCell);
        if (p < beg || p >= end)
          continue;
        const uptr s = MemToShadowImpl::Apply<Mapping>(p);
        u32 *const m = MemToMetaImpl::Apply<Mapping>(p);
        const uptr r = ShadowToMemImpl::Apply<Mapping>(s);
        Printf("  addr=0x%zx: shadow=0x%zx meta=%p reverse=0x%zx\n", p, s, m,
               r);
        CHECK(IsAppMemImpl::Apply<Mapping>(p));
        if (!broken<Mapping>(kBrokenMapping))
          CHECK(IsShadowMemImpl::Apply<Mapping>(s));
        CHECK(IsMetaMemImpl::Apply<Mapping>(reinterpret_cast<uptr>(m)));
        CHECK_EQ(p, RestoreAddrImpl::Apply<Mapping>(CompressAddr(p)));
        if (!broken<Mapping>(kBrokenReverseMapping))
          CHECK_EQ(p, r);
        if (prev && !broken<Mapping>(kBrokenLinearity)) {
          // Ensure that shadow and meta mappings are linear within a single
          // user range. Lots of code that processes memory ranges assumes it.
          const uptr prev_s = MemToShadowImpl::Apply<Mapping>(prev);
          u32 *const prev_m = MemToMetaImpl::Apply<Mapping>(prev);
          CHECK_EQ(s - prev_s, (p - prev) * kShadowMultiplier);
          CHECK_EQ(m - prev_m, (p - prev) / kMetaShadowCell);
        }
        prev = p;
      }
    }
  }
};

TEST(Shadow, AllMappings) { ForEachMapping<MappingTest>(); }

}  // namespace __tsan
