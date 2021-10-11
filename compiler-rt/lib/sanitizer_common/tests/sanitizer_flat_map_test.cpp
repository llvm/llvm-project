//===-- sanitizer_flat_map_test.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_flat_map.h"

#include "gtest/gtest.h"
#include "sanitizer_common/tests/sanitizer_pthread_wrappers.h"

using namespace __sanitizer;

namespace {
struct TestMapUnmapCallback1 {
  static int map_count, unmap_count;
  void OnMap(uptr p, uptr size) const { map_count++; }
  void OnUnmap(uptr p, uptr size) const { unmap_count++; }
};
int TestMapUnmapCallback1::map_count;
int TestMapUnmapCallback1::unmap_count;

TEST(FlatMapTest, TwoLevelByteMap) {
  const u64 kSize1 = 1 << 6, kSize2 = 1 << 12;
  const u64 n = kSize1 * kSize2;
  TwoLevelByteMap<kSize1, kSize2> m;
  m.Init();
  for (u64 i = 0; i < n; i += 7) {
    m.set(i, (i % 100) + 1);
  }
  for (u64 j = 0; j < n; j++) {
    if (j % 7)
      EXPECT_EQ(m[j], 0);
    else
      EXPECT_EQ(m[j], (j % 100) + 1);
  }

  m.TestOnlyUnmap();
}

template <typename AddressSpaceView>
using TestByteMapASVT =
    TwoLevelByteMap<1 << 12, 1 << 13, AddressSpaceView, TestMapUnmapCallback1>;
using TestByteMap = TestByteMapASVT<LocalAddressSpaceView>;

struct TestByteMapParam {
  TestByteMap *m;
  size_t shard;
  size_t num_shards;
};

static void *TwoLevelByteMapUserThread(void *param) {
  TestByteMapParam *p = (TestByteMapParam *)param;
  for (size_t i = p->shard; i < p->m->size(); i += p->num_shards) {
    size_t val = (i % 100) + 1;
    p->m->set(i, val);
    EXPECT_EQ((*p->m)[i], val);
  }
  return 0;
}

TEST(FlatMapTest, ThreadedTwoLevelByteMap) {
  TestByteMap m;
  m.Init();
  TestMapUnmapCallback1::map_count = 0;
  TestMapUnmapCallback1::unmap_count = 0;
  static const int kNumThreads = 4;
  pthread_t t[kNumThreads];
  TestByteMapParam p[kNumThreads];
  for (int i = 0; i < kNumThreads; i++) {
    p[i].m = &m;
    p[i].shard = i;
    p[i].num_shards = kNumThreads;
    PTHREAD_CREATE(&t[i], 0, TwoLevelByteMapUserThread, &p[i]);
  }
  for (int i = 0; i < kNumThreads; i++) {
    PTHREAD_JOIN(t[i], 0);
  }
  EXPECT_EQ((uptr)TestMapUnmapCallback1::map_count, m.size1());
  EXPECT_EQ((uptr)TestMapUnmapCallback1::unmap_count, 0UL);
  m.TestOnlyUnmap();
  EXPECT_EQ((uptr)TestMapUnmapCallback1::map_count, m.size1());
  EXPECT_EQ((uptr)TestMapUnmapCallback1::unmap_count, m.size1());
}

}  // namespace
