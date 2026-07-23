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
// class unsynchronized_pool_resource

#include <memory_resource>
#include <cassert>
#include <new>
#include <type_traits>

#include "count_new.h"

class assert_on_compare : public std::pmr::memory_resource {
  void* do_allocate(std::size_t, size_t) override {
    assert(false);
    return nullptr;
  }

  void do_deallocate(void*, std::size_t, size_t) override { assert(false); }

  bool do_is_equal(const std::pmr::memory_resource&) const noexcept override {
    assert(false);
    return true;
  }
};

template <class PoolResource>
void test() {
  // Same type
  {
    PoolResource pr1;
    PoolResource pr2;
    assert(pr1 == pr1);
    assert(pr1 != pr2);

    std::pmr::memory_resource& mr1 = pr1;
    std::pmr::memory_resource& mr2 = pr2;
    assert(mr1 == mr1);
    assert(mr1 != mr2);
    assert(mr1 == pr1);
    assert(pr1 == mr1);
    assert(mr1 != pr2);
    assert(pr2 != mr1);
  }
  // Different types
  {
    PoolResource pr1;
    std::pmr::memory_resource& mr1 = pr1;
    assert_on_compare c;
    std::pmr::memory_resource& mr2 = c;
    assert(mr1 != mr2);
    assert(!(mr1 == mr2));
  }
}

int main(int, char**) {
  test<std::pmr::synchronized_pool_resource>();
  test<std::pmr::unsynchronized_pool_resource>();

  return 0;
}
