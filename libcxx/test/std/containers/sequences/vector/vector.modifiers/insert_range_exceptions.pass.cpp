//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-exceptions

// <vector>

// template<container-compatible-range<T> R>
//   constexpr iterator insert_range(const_iterator position, R&& rg); // C++23

#include <cassert>
#include <stdexcept>
#include <vector>

#include "../../insert_range_sequence_containers.h"
#include "test_allocator.h"

int main(int, char**) {
  test_insert_range_exception_safety_throwing_copy<std::vector>();
  test_insert_range_exception_safety_throwing_allocator<std::vector, int>();

  { // Attempt to insert a range at the beginning of a vector that would exceed its maximum possible size
    std::vector<int, limited_allocator<int, 10> > v(8, 42);
    int a[] = {1, 2, 3};
    try {
      v.insert_range(v.begin(), a);
      assert(false);
    } catch (const std::length_error&) {
      assert(v.size() == 8);
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == 42);
    }
  }
  { // Attempt to insert a range at the end of a vector that would exceed its maximum possible size
    std::vector<int, limited_allocator<int, 10> > v(8, 42);
    int a[] = {1, 2, 3};
    try {
      v.insert_range(v.end(), a);
      assert(false);
    } catch (const std::length_error&) {
      assert(v.size() == 8);
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == 42);
    }
  }
  { // Attempt to insert a range in the middle of a vector that would exceed its maximum possible size
    std::vector<int, limited_allocator<int, 10> > v(10, 42);
    int a[] = {1, 2, 3};
    try {
      v.insert_range(v.begin() + v.size() / 2, a);
      assert(false);
    } catch (const std::length_error&) {
      assert(v.size() == 10);
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == 42);
    }
  }

  return 0;
}
