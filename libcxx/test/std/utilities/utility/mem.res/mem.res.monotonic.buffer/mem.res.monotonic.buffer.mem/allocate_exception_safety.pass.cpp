//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: no-exceptions
// XFAIL: availability-pmr-missing

// <memory_resource>

// class monotonic_buffer_resource

#include <memory_resource>
#include <cassert>

#include "count_new.h"
#include "test_macros.h"

struct repointable_resource : public std::pmr::memory_resource {
  std::pmr::memory_resource* which;

  explicit repointable_resource(std::pmr::memory_resource* res) : which(res) {}

private:
  void* do_allocate(std::size_t size, size_t align) override { return which->allocate(size, align); }

  void do_deallocate(void* p, std::size_t size, size_t align) override { return which->deallocate(p, size, align); }

  bool do_is_equal(std::pmr::memory_resource const& rhs) const noexcept override { return which->is_equal(rhs); }
};

void test_exception_safety() {
  globalMemCounter.reset();
  auto upstream = repointable_resource(std::pmr::new_delete_resource());
  alignas(16) char buffer[64];
  auto mono1                    = std::pmr::monotonic_buffer_resource(buffer, sizeof buffer, &upstream);
  std::pmr::memory_resource& r1 = mono1;

  void* res = r1.allocate(64, 16);
  assert(res == buffer);
  assert(globalMemCounter.checkNewCalledEq(0));

  res = r1.allocate(64, 16);
  assert(res != buffer);
  assert(globalMemCounter.checkNewCalledEq(1));
  assert(globalMemCounter.checkDeleteCalledEq(0));
  const std::size_t last_new_size = globalMemCounter.last_new_size;

  upstream.which = std::pmr::null_memory_resource();
  try {
    res = r1.allocate(last_new_size, 16);
    assert(false);
  } catch (const std::bad_alloc&) {
    // we expect this
  }
  assert(globalMemCounter.checkNewCalledEq(1));
  assert(globalMemCounter.checkDeleteCalledEq(0));

  upstream.which = std::pmr::new_delete_resource();
  res            = r1.allocate(last_new_size, 16);
  assert(res != buffer);
  assert(globalMemCounter.checkNewCalledEq(2));
  assert(globalMemCounter.checkDeleteCalledEq(0));

  mono1.release();
  assert(globalMemCounter.checkNewCalledEq(2));
  assert(globalMemCounter.checkDeleteCalledEq(2));
}

int main(int, char**) {
#if TEST_SUPPORTS_LIBRARY_INTERNAL_ALLOCATIONS && !defined(DISABLE_NEW_COUNT)
  test_exception_safety();
#endif

  return 0;
}
