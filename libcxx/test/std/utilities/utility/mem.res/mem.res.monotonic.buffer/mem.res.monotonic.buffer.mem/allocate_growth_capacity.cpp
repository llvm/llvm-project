//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// TODO: Change to XFAIL once https://github.com/llvm/llvm-project/issues/40340 is fixed
// UNSUPPORTED: availability-pmr-missing

// <memory_resource>

// class monotonic_buffer_resource

#include <memory_resource>
#include <cassert>

#include "count_new.h"
#include "test_macros.h"

// LWG 3143: https://cplusplus.github.io/LWG/issue3143
void test_growth_capacity() {
  // mem.res.monotonic.buffer 1.3
  // Each additional buffer is larger than the previous one

  constexpr auto foot_size{4 * sizeof(void*)};

  globalMemCounter.reset();
  std::pmr::monotonic_buffer_resource mono1(100, std::pmr::new_delete_resource());
  std::pmr::memory_resource& r1 = mono1;

  assert(globalMemCounter.checkNewCalledEq(0));
  std::size_t next_buffer_size = 100;
  void* ret                    = r1.allocate(10, 1);
  assert(ret != nullptr);
  assert(globalMemCounter.checkNewCalledEq(1));
  assert(globalMemCounter.last_new_size >= next_buffer_size);
  next_buffer_size = globalMemCounter.last_new_size;

  int new_called = 1;
  while (new_called < 5) {
    ret = r1.allocate(10, 1);
    if (globalMemCounter.new_called > new_called) {
      assert(globalMemCounter.new_called == new_called + 1);
      next_buffer_size = next_buffer_size * 2 - foot_size;
      assert(globalMemCounter.last_new_size == next_buffer_size);
      new_called += 1;
    }
  }
}

int main(int, char**) {
#if TEST_SUPPORTS_LIBRARY_INTERNAL_ALLOCATIONS && !defined(DISABLE_NEW_COUNT)
  test_growth_capacity();
#endif

  return 0;
}
