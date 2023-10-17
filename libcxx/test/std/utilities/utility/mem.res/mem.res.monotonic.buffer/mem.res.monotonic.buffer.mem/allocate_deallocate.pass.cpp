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

int main(int, char**) {
  {
    globalMemCounter.reset();

    auto mono1                    = std::pmr::monotonic_buffer_resource(std::pmr::new_delete_resource());
    std::pmr::memory_resource& r1 = mono1;

    void* ret = r1.allocate(50);
    assert(ret);
    ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkNewCalledGreaterThan(0));
    assert(globalMemCounter.checkDeleteCalledEq(0));

    // mem.res.monotonic.buffer 1.2
    // A call to deallocate has no effect, thus the amount of memory
    // consumed increases monotonically until the resource is destroyed.
    // Check that deallocate is a no-op
    r1.deallocate(ret, 50);
    assert(globalMemCounter.checkDeleteCalledEq(0));

    mono1.release();
    ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkDeleteCalledEq(1));
    assert(globalMemCounter.checkOutstandingNewEq(0));

    globalMemCounter.reset();

    ret = r1.allocate(500);
    assert(ret);
    ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkNewCalledGreaterThan(0));
    assert(globalMemCounter.checkDeleteCalledEq(0));

    // Check that the destructor calls release()
  }
  ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkDeleteCalledEq(1));
  assert(globalMemCounter.checkOutstandingNewEq(0));

  return 0;
}
