//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// void push_back(const value_type& x);
//
// If no reallocation happens, then references, pointers, and iterators before
// the insertion point remain valid but those at or after the insertion point,
// including the past-the-end iterator, are invalidated.

// REQUIRES: has-unix-headers, libcpp-has-abi-bounded-iterators-in-vector
// UNSUPPORTED: c++03
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <vector>
#include <cassert>
#include <cstddef>

#include "check_assertion.h"

int main(int, char**) {
  std::vector<int> vec;
  vec.reserve(4);
  std::size_t old_capacity = vec.capacity();
  assert(old_capacity >= 4);

  vec.push_back(0);
  vec.push_back(1);
  vec.push_back(2);
  auto it = vec.begin();
  vec.push_back(3);
  assert(vec.capacity() == old_capacity);

  // The capacity did not change, so the iterator remains valid and can reach the new element.
  assert(*it == 0);
  assert(*(it + 1) == 1);
  assert(*(it + 2) == 2);
  assert(*(it + 3) == 3);

  while (vec.capacity() == old_capacity) {
    vec.push_back(42);
  }
  TEST_LIBCPP_ASSERT_FAILURE(
      *(it + old_capacity), "__bounded_iter::operator*: Attempt to dereference an iterator at the end");
  // Unfortunately, the bounded iterator does not detect that it's been invalidated and will still allow attempts to
  // dereference elements 0 to 3 (even though they refer to memory that's been reallocated).

  return 0;
}
