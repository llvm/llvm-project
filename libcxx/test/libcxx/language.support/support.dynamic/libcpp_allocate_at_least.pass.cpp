//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// Check that __libcpp_allocate_at_least works as expected

#include <__memory/is_sufficiently_aligned.h>
#include <__new/allocate.h>
#include <cassert>
#include <limits>
#include <new>

#include "test_macros.h"

int new_handler_called = 0;

void my_new_handler() {
  ++new_handler_called;
  std::set_new_handler(nullptr);
}

int main(int, char**) {
  { // Check that a simple call works as expected
    auto result = std::__libcpp_allocate_at_least<char>(std::__element_count(1));
    assert(result.ptr);
    assert(result.count >= 1);
#if defined(_LIBCPP_AVAILABILITY_HAS_NEW_AT_LEAST) && (defined(__APPLE__) || defined(__FreeBSD__))
    // Check that on platforms where we have a better implementation than the generic one we will always get more than
    // one byte. This is technically not guaranteed, but allocators are in general not capable of allocating individual
    // bytes efficiently.
    assert(result.count > 1);
#endif
    operator delete(result.ptr, result.count);
  }

  { // Check that any size calculations are done correctly
    auto result = std::__libcpp_allocate_at_least<int>(std::__element_count(3));
    assert(result.ptr);
    assert(result.count >= 3);
    operator delete(result.ptr, result.count);
  }

#ifndef TEST_HAS_NO_ALIGNED_ALLOCATION
  { // Check that the aligned version is called with when using an alignment
    // that's larger than the default new alignment
    auto result = std::__libcpp_allocate_at_least<char>(std::__element_count(1), __STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2);
    assert(result.ptr);
    assert(std::__is_sufficiently_aligned<__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2>(result.ptr));
    assert(result.count >= 1);
#  if defined(_LIBCPP_AVAILABILITY_HAS_NEW_AT_LEAST) && (defined(__APPLE__) || defined(__FreeBSD__))
    // Check that on platforms where we have a better implementation than the generic one we will always get more than
    // one byte. This is technically not guaranteed, but allocators are in general not capable of allocating individual
    // bytes efficiently.
    assert(result.count > 1);
#  endif
    operator delete(result.ptr, result.count, std::align_val_t(__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2));
  }

  { // Check that any size calculations are done correctly
    auto result = std::__libcpp_allocate_at_least<int>(std::__element_count(3), __STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2);
    assert(result.ptr);
    assert(result.count >= 3);
    operator delete(result.ptr, result.count);
  }
#endif // TEST_HAS_NO_ALIGNED_ALLOCATION

  { // Test that the new handler is called if allocation fails
#ifndef TEST_HAS_NO_EXCEPTIONS
    new_handler_called = 0;
    std::set_new_handler(my_new_handler);
    try {
      (void)std::__libcpp_allocate_at_least<char>(std::__element_count(std::numeric_limits<std::size_t>::max()));
      assert(false);
    } catch (std::bad_alloc const&) {
      assert(new_handler_called == 1);
    } catch (...) {
      assert(false);
    }
#endif
  }

  { // Test that the new handler is called if allocation fails (aligned overload)
#if !defined(TEST_HAS_NO_EXCEPTIONS) && !defined(TEST_HAS_NO_ALIGNED_ALLOCATION)
    new_handler_called = 0;
    std::set_new_handler(my_new_handler);
    try {
      (void)std::__libcpp_allocate_at_least<char>(
          std::__element_count(std::numeric_limits<std::size_t>::max()), __STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2);
      assert(false);
    } catch (std::bad_alloc const&) {
      assert(new_handler_called == 1);
    } catch (...) {
      assert(false);
    }
#endif
  }

  return 0;
}
