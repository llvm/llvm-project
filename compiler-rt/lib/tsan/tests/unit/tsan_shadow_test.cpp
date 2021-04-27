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
