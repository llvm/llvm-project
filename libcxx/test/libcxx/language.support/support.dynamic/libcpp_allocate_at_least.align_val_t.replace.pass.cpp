//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// Check that __libcpp_allocate_at_least falls back to a user-provided operator new if provided.

#include <__memory/is_sufficiently_aligned.h>
#include <__new/allocate.h>
#include <cassert>
#include <cstdlib>
#include <new>

#include "test_macros.h"

int new_called    = 0;
int delete_called = 0;

alignas(__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2) char data[32];

void* operator new(std::size_t, std::align_val_t) {
  ++new_called;
  return data;
}

void operator delete(void*, std::align_val_t) noexcept { ++delete_called; }

int main(int, char**) {
  { // Check that a simple call works as expected
    auto result = std::__libcpp_allocate_at_least<char>(std::__element_count(1));
    assert(new_called == 0);
    operator delete(result.ptr, result.count);
    assert(delete_called == 0);
  }

  // operator new(size_t, align_val_t) isn't overridden, so we still use the special implementation.
#ifndef TEST_HAS_NO_ALIGNED_ALLOCATION
  { // Check that the aligned version is called with when using an alignment
    // that's larger than the default new alignment
    auto result = std::__libcpp_allocate_at_least<char>(std::__element_count(1), __STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2);
    assert(new_called == 1);
    new_called = 0;
    operator delete(result.ptr, result.count, std::align_val_t(__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2));
    assert(delete_called == 1);
    delete_called = 0;
  }
#endif // TEST_HAS_NO_ALIGNED_ALLOCATION

  return 0;
}
