//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17
// UNSUPPORTED: availability-pmr-missing

// <memory_resource>

// class monotonic_buffer_resource

// This test checks the behavior required by LWG3120.

#include <memory_resource>
#include <cassert>

#include "count_new.h"
#include "test_macros.h"

int main(int, char**) {
  {
    {
      // When ctor with a next buffer size. After release(), check whether the next buffer size has been reset after release()
      constexpr size_t expect_next_buffer_size = 512;
      std::pmr::monotonic_buffer_resource mr{nullptr, expect_next_buffer_size, std::pmr::new_delete_resource()};

      for (int i = 0; i < 100; ++i) {
        (void)mr.allocate(1);
        mr.release();
        ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkLastNewSizeGe(expect_next_buffer_size));
      }
    }
    {
      // Check whether the offset of the initial buffer has been reset after release()
      constexpr size_t buffer_size = 100;
      char buffer[buffer_size];
      std::pmr::monotonic_buffer_resource mr{buffer, buffer_size, std::pmr::null_memory_resource()};

      mr.release();
      auto expect_mem_start = mr.allocate(60);
      mr.release();
      auto ths_mem_start = mr.allocate(60);
      assert(expect_mem_start == ths_mem_start);
    }
  }
  return 0;
}
