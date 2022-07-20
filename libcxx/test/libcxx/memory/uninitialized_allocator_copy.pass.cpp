//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// ensure that __uninitialized_allocator_copy calls the proper construct and destruct functions

#include <algorithm>
#include <memory>

#include "test_allocator.h"

template <class T>
class construct_counting_allocator {
public:
  using value_type = T;

  int* constructed_count_;
  int* max_constructed_count_;

  construct_counting_allocator(int* constructed_count, int* max_constructed_count)
      : constructed_count_(constructed_count), max_constructed_count_(max_constructed_count) {}

  template <class... Args>
  void construct(T* ptr, Args&&... args) {
    ::new (static_cast<void*>(ptr)) T(args...);
    ++*constructed_count_;
    *max_constructed_count_ = std::max(*max_constructed_count_, *constructed_count_);
  }

  void destroy(T* ptr) {
    --*constructed_count_;
    ptr->~T();
  }
};

int throw_if_zero = 15;

struct ThrowSometimes {
  ThrowSometimes() = default;
  ThrowSometimes(const ThrowSometimes&) {
    if (--throw_if_zero == 0)
      throw 1;
  }
};

int main(int, char**) {
  int constructed_count     = 0;
  int max_constructed_count = 0;
  construct_counting_allocator<ThrowSometimes> alloc(&constructed_count, &max_constructed_count);
  ThrowSometimes in[20];
  TEST_ALIGNAS_TYPE(ThrowSometimes) char out[sizeof(ThrowSometimes) * 20];
  try {
    std::__uninitialized_allocator_copy(
        alloc, std::begin(in), std::end(in), reinterpret_cast<ThrowSometimes*>(std::begin(out)));
  } catch (...) {
  }

  assert(constructed_count == 0);
  assert(max_constructed_count == 14);
}
