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

#include "test_macros.h"
#include "count_new.h"

int main(int, char**) {
  globalMemCounter.reset();
  auto mono1                    = std::pmr::monotonic_buffer_resource(1024, std::pmr::new_delete_resource());
  std::pmr::memory_resource& r1 = mono1;

  constexpr size_t big_alignment = 8 * alignof(std::max_align_t);
  static_assert(big_alignment > 4);

  void* ret = r1.allocate(2048, big_alignment);
  assert(ret != nullptr);
  ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkNewCalledEq(1));
  ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkAlignedNewCalledEq(1));
  ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkLastNewSizeGe(2048));
  ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkLastNewAlignEq(big_alignment));

  // Check that a single highly aligned allocation request doesn't
  // permanently "poison" the resource to allocate only super-aligned
  // blocks of memory.
  ret = r1.allocate(globalMemCounter.last_new_size, 4);
  assert(ret != nullptr);
  ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkNewCalledEq(2));
  ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkAlignedNewCalledEq(1));

  return 0;
}
