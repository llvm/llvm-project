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

// class synchronized_pool_resource

#include <memory_resource>
#include <cassert>

#include "count_new.h"
#include "test_macros.h"

int main(int, char**) {
  globalMemCounter.reset();
  {
    auto sync1                    = std::pmr::synchronized_pool_resource(std::pmr::new_delete_resource());
    std::pmr::memory_resource& r1 = sync1;

    void* ret = r1.allocate(50);
    assert(ret);
    ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkNewCalledGreaterThan(0));
    ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkDeleteCalledEq(0));

    r1.deallocate(ret, 50);
    sync1.release();
    ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkDeleteCalledGreaterThan(0));
    assert(globalMemCounter.checkOutstandingNewEq(0));

    globalMemCounter.reset();

    ret = r1.allocate(500);
    assert(ret);
    ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkNewCalledGreaterThan(0));
    assert(globalMemCounter.checkDeleteCalledEq(0));

    // Check that the destructor calls release()
  }
  ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkDeleteCalledGreaterThan(0));
  assert(globalMemCounter.checkOutstandingNewEq(0));

  return 0;
}
