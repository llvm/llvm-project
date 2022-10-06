//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14|15}}
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{11.0|12.0}}

// <memory_resource>

// class monotonic_buffer_resource

#include <memory_resource>
#include <cassert>

#include "count_new.h"
#include "test_macros.h"

int main(int, char**) {
  globalMemCounter.reset();
  char buffer[100];
  auto mono1 = std::pmr::monotonic_buffer_resource(buffer, sizeof buffer, std::pmr::new_delete_resource());
  std::pmr::memory_resource& r1 = mono1;

  // Check that construction with a buffer does not allocate anything from the upstream
  assert(globalMemCounter.checkNewCalledEq(0));

  // Check that an allocation that fits in the buffer does not allocate anything from the upstream
  void* ret = r1.allocate(50);
  assert(ret);
  assert(globalMemCounter.checkNewCalledEq(0));

  // Check a second allocation
  ret = r1.allocate(20);
  assert(ret);
  assert(globalMemCounter.checkNewCalledEq(0));

  r1.deallocate(ret, 50);
  assert(globalMemCounter.checkDeleteCalledEq(0));

  // Check an allocation that doesn't fit in the original buffer
  ret = r1.allocate(50);
  assert(ret);
  ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkNewCalledEq(1));

  r1.deallocate(ret, 50);
  assert(globalMemCounter.checkDeleteCalledEq(0));

  mono1.release();
  ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkDeleteCalledEq(1));
  assert(globalMemCounter.checkOutstandingNewEq(0));

  return 0;
}
