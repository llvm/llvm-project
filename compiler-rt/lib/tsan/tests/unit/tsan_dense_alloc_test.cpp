//===-- tsan_dense_alloc_test.cpp -----------------------------------------===//
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
#include "tsan_dense_alloc.h"
#include "tsan_rtl.h"
#include "tsan_mman.h"
#include "gtest/gtest.h"

#include <stdlib.h>
#include <stdint.h>
#include <map>

namespace __tsan {

TEST(DenseSlabAlloc, Basic) {
  typedef DenseSlabAlloc<int, 128, 128> Alloc;
  typedef Alloc::Cache Cache;
  typedef Alloc::IndexT IndexT;
  const int N = 1000;

  Alloc alloc;
  Cache cache;
  alloc.InitCache(&cache);

  IndexT blocks[N];
  for (int ntry = 0; ntry < 3; ntry++) {
    for (int i = 0; i < N; i++) {
      IndexT idx = alloc.Alloc(&cache);
      blocks[i] = idx;
      EXPECT_NE(idx, 0U);
      int *v = alloc.Map(idx);
      *v = i;
    }

    for (int i = 0; i < N; i++) {
      IndexT idx = blocks[i];
      int *v = alloc.Map(idx);
      EXPECT_EQ(*v, i);
      alloc.Free(&cache, idx);
    }

    alloc.FlushCache(&cache);
  }
}

}  // namespace __tsan
