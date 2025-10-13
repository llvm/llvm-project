//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// TODO: Change to XFAIL once https://llvm.org/PR40995 is fixed
// UNSUPPORTED: availability-pmr-missing

// <memory_resource>

// class monotonic_buffer_resource

#include <memory_resource>
#include <cassert>

#include "count_new.h"
#include "test_macros.h"

int main(int, char**) {
  globalMemCounter.reset();
  {
    char buffer[100];
    auto mono1                    = std::pmr::monotonic_buffer_resource(buffer, 0, std::pmr::new_delete_resource());
    std::pmr::memory_resource& r1 = mono1;

    void* ret = r1.allocate(1, 1);
    assert(ret != nullptr);
    ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkNewCalledEq(1));
  }
  ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkDeleteCalledEq(1));
  assert(globalMemCounter.checkOutstandingNewEq(0));

  globalMemCounter.reset();
  {
    auto mono1                    = std::pmr::monotonic_buffer_resource(nullptr, 0, std::pmr::new_delete_resource());
    std::pmr::memory_resource& r1 = mono1;

    void* ret = r1.allocate(1, 1);
    assert(ret != nullptr);
    ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkNewCalledEq(1));
  }
  ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkDeleteCalledEq(1));
  assert(globalMemCounter.checkOutstandingNewEq(0));

  return 0;
}
