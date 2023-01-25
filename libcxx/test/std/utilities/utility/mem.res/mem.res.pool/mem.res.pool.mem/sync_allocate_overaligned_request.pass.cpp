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

// class synchronized_pool_resource

#include <memory_resource>
#include <cassert>
#include <memory> // std::align

#include "count_new.h"
#include "test_macros.h"

bool is_aligned_to(void* p, size_t alignment) {
  void* p2     = p;
  size_t space = 1;
  void* result = std::align(alignment, 1, p2, space);
  return (result == p);
}

int main(int, char**) {
  globalMemCounter.reset();
  std::pmr::pool_options opts{1, 1024};
  std::pmr::synchronized_pool_resource sync1(opts, std::pmr::new_delete_resource());
  std::pmr::memory_resource& r1 = sync1;

  constexpr size_t big_alignment = 8 * alignof(std::max_align_t);
  static_assert(big_alignment > 4);

  assert(globalMemCounter.checkNewCalledEq(0));

  void* ret = r1.allocate(2048, big_alignment);
  assert(ret != nullptr);
  assert(is_aligned_to(ret, big_alignment));
  ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkNewCalledGreaterThan(0));

  ret = r1.allocate(16, 4);
  assert(ret != nullptr);
  assert(is_aligned_to(ret, 4));
  ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkNewCalledGreaterThan(1));

  return 0;
}
