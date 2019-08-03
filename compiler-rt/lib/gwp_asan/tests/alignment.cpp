//===-- alignment.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/tests/harness.h"

TEST_F(DefaultGuardedPoolAllocator, BasicAllocation) {
  std::vector<std::pair<int, int>> AllocSizeToAlignment = {
      {1, 1},   {2, 2},   {3, 4},       {4, 4},       {5, 8},   {7, 8},
      {8, 8},   {9, 16},  {15, 16},     {16, 16},     {17, 16}, {31, 16},
      {32, 16}, {33, 16}, {4095, 4096}, {4096, 4096},
  };

  for (const auto &KV : AllocSizeToAlignment) {
    void *Ptr = GPA.allocate(KV.first);
    EXPECT_NE(nullptr, Ptr);

    // Check the alignment of the pointer is as expected.
    EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(Ptr) % KV.second);

    GPA.deallocate(Ptr);
  }
}
