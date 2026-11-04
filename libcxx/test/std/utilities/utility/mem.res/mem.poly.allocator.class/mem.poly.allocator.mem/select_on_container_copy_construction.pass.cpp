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

// template <class T> class polymorphic_allocator

// polymorphic_allocator
// polymorphic_allocator<T>::select_on_container_copy_construction() const

#include <cassert>
#include <cstddef>
#include <memory_resource>
#include <new>

#include "test_macros.h"

struct resource : std::pmr::memory_resource {
  void* do_allocate(size_t, size_t) override { TEST_THROW(std::bad_alloc()); }
  void do_deallocate(void*, size_t, size_t) override { assert(false); }
  bool do_is_equal(const std::pmr::memory_resource&) const noexcept override { return false; }
};

int main(int, char**) {
  typedef std::pmr::polymorphic_allocator<void> A;
  {
    A const a;
    ASSERT_SAME_TYPE(decltype(a.select_on_container_copy_construction()), A);
  }
  {
    resource res;
    A const a(&res);
    assert(a.resource() == &res);
    A const other = a.select_on_container_copy_construction();
    assert(other.resource() == std::pmr::get_default_resource());
    assert(a.resource() == &res);
  }

  return 0;
}
