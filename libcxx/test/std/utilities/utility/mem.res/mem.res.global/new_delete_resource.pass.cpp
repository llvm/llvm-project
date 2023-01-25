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

// memory_resource *new_delete_resource()

#include <memory_resource>
#include <cassert>
#include <type_traits>

#include "count_new.h"
#include "test_macros.h"

class assert_on_compare : public std::pmr::memory_resource {
  void* do_allocate(size_t, size_t) override {
    assert(false);
    return nullptr;
  }

  void do_deallocate(void*, size_t, size_t) override { assert(false); }

  bool do_is_equal(const std::pmr::memory_resource&) const noexcept override {
    assert(false);
    return true;
  }
};

void test_return() {
  { ASSERT_SAME_TYPE(decltype(std::pmr::new_delete_resource()), std::pmr::memory_resource*); }
  // assert not null
  { assert(std::pmr::new_delete_resource()); }
  // assert same return value
  { assert(std::pmr::new_delete_resource() == std::pmr::new_delete_resource()); }
}

void test_equality() {
  // Same object
  {
    std::pmr::memory_resource& r1 = *std::pmr::new_delete_resource();
    std::pmr::memory_resource& r2 = *std::pmr::new_delete_resource();
    // check both calls returned the same object
    assert(&r1 == &r2);
    // check for proper equality semantics
    assert(r1 == r2);
    assert(r2 == r1);
    assert(!(r1 != r2));
    assert(!(r2 != r1));
  }
  // Different types
  {
    std::pmr::memory_resource& r1 = *std::pmr::new_delete_resource();
    assert_on_compare c;
    std::pmr::memory_resource& r2 = c;
    assert(r1 != r2);
    assert(!(r1 == r2));
  }
}

void test_allocate_deallocate() {
  std::pmr::memory_resource& r1 = *std::pmr::new_delete_resource();

  globalMemCounter.reset();

  void* ret = r1.allocate(50);
  assert(ret);
  ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkOutstandingNewEq(1));
  ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkLastNewSizeEq(50));

  r1.deallocate(ret, 1);
  assert(globalMemCounter.checkOutstandingNewEq(0));
  ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkDeleteCalledEq(1));
}

int main(int, char**) {
  ASSERT_NOEXCEPT(std::pmr::new_delete_resource());
  test_return();
  test_equality();
  test_allocate_deallocate();

  return 0;
}
