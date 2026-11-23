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

// memory_resource *null_memory_resource()

#include <memory_resource>
#include <cassert>
#include <cstddef> // size_t
#include <new>
#include <type_traits>

#include "count_new.h"
#include "test_macros.h"

struct assert_on_compare : public std::pmr::memory_resource {
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

void test_return() {
  { ASSERT_SAME_TYPE(decltype(std::pmr::null_memory_resource()), std::pmr::memory_resource*); }
  // Test that the returned value is not null
  { assert(std::pmr::null_memory_resource()); }
  // Test the same value is returned by repeated calls.
  { assert(std::pmr::null_memory_resource() == std::pmr::null_memory_resource()); }
}

void test_equality() {
  // Same object
  {
    std::pmr::memory_resource& r1 = *std::pmr::null_memory_resource();
    std::pmr::memory_resource& r2 = *std::pmr::null_memory_resource();
    // check both calls returned the same object
    assert(&r1 == &r2);
    // check for proper equality semantics
    assert(r1 == r2);
    assert(r2 == r1);
    assert(!(r1 != r2));
    assert(!(r2 != r1));
    // check the is_equal method
    assert(r1.is_equal(r2));
    assert(r2.is_equal(r1));
  }
  // Different types
  {
    std::pmr::memory_resource& r1 = *std::pmr::null_memory_resource();
    assert_on_compare c;
    std::pmr::memory_resource& r2 = c;
    assert(r1 != r2);
    assert(!(r1 == r2));
    assert(!r1.is_equal(r2));
  }
}

void test_allocate() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  DisableAllocationGuard g; // null_memory_resource shouldn't allocate.
  try {
    (void)std::pmr::null_memory_resource()->allocate(1);
    assert(false);
  } catch (std::bad_alloc const&) {
    // do nothing
  } catch (...) {
    assert(false);
  }
#endif
}

void test_deallocate() {
  globalMemCounter.reset();

  int x = 42;
  std::pmr::null_memory_resource()->deallocate(&x, 0);

  assert(globalMemCounter.checkDeleteCalledEq(0));
  assert(globalMemCounter.checkDeleteArrayCalledEq(0));
}

int main(int, char**) {
  test_return();
  test_equality();
  test_allocate();
  test_deallocate();

  return 0;
}
