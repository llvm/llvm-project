//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory_resource>

// Test hardening assertions for std::pmr::polymorphic_allocator.

// REQUIRES: can-test-hardening-assertions-extensive
// UNSUPPORTED: c++03, c++11, c++14

// We're testing nullptr assertions
// ADDITIONAL_COMPILE_FLAGS: -Wno-nonnull -Wno-non-power-of-two-alignment

#include <cassert>
#include <memory_resource>

#include "check_assertion.h"

struct my_memory_resource : std::pmr::memory_resource {
  void* do_allocate(std::size_t, std::size_t) override { assert(false); }
  void do_deallocate(void*, std::size_t, std::size_t) override { assert(false); }
  bool do_is_equal(const std::pmr::memory_resource&) const noexcept override { return false; }
};

int main(int, char**) {
  TEST_LIBCPP_ASSERT_FAILURE(
      std::pmr::polymorphic_allocator<int>(nullptr), "Attempted to pass a nullptr resource to polymorphic_alloator");

  TEST_LIBCPP_ASSERT_FAILURE(my_memory_resource().allocate(0, 0), "The alignment has to be a power of two");
  TEST_LIBCPP_ASSERT_FAILURE(my_memory_resource().allocate(0, 33), "The alignment has to be a power of two");

  return 0;
}
