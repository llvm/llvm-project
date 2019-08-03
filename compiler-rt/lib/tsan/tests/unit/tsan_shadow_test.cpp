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
  CHECK_EQ((uptr)data % kShadowSize, 0);
  uptr s0 = MemToShadow((uptr)&data[0]);
  CHECK_EQ(s0 % kShadowSize, 0);
  for (unsigned i = 1; i < kShadowCell; i++)
    CHECK_EQ(s0, MemToShadow((uptr)&data[i]));
  for (unsigned i = kShadowCell; i < 2*kShadowCell; i++)
    CHECK_EQ(s0 + kShadowSize*kShadowCnt, MemToShadow((uptr)&data[i]));
  for (unsigned i = 2*kShadowCell; i < 3*kShadowCell; i++)
    CHECK_EQ(s0 + 2*kShadowSize*kShadowCnt, MemToShadow((uptr)&data[i]));
}

}  // namespace __tsan
