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
// vector<bool>

// template<container-compatible-range<T> R>
//   constexpr iterator insert_range(const_iterator position, R&& rg); // C++23

#include <cassert>
#include <stdexcept>
#include <vector>

#include "../insert_range_sequence_containers.h"
#include "test_allocator.h"

int main(int, char**) {
  // Note: `test_insert_range_exception_safety_throwing_copy` doesn't apply because copying booleans cannot throw.
  test_insert_range_exception_safety_throwing_allocator<std::vector, bool>();

  { // Attempt to insert a range at the end of a vector<bool> that would exceed its maximum possible size
    std::vector<bool, limited_allocator<bool, 10> > v;
    v.resize(v.max_size() - 2, true);
    bool a[] = {true, false, true};
    try {
      v.insert_range(v.end(), a);
      assert(false);
    } catch (const std::length_error&) {
      assert(v.size() == v.max_size() - 2);
      for (std::size_t i = 0; i < v.size(); ++i)
        assert(v[i] == true);
    }
  }
  { // Attempt to insert a range at the beginning of a vector<bool> that would exceed its maximum possible size
    std::vector<bool, limited_allocator<bool, 10> > v;
    v.resize(v.max_size() - 2, false);
    bool a[] = {true, false, true};
    try {
      v.insert_range(v.begin(), a);
      assert(false);
    } catch (const std::length_error&) {
      assert(v.size() == v.max_size() - 2);
      for (std::size_t i = 0; i < v.size(); ++i)
        assert(v[i] == false);
    }
  }
  { // Attempt to insert a range in the middle of a vector<bool> that would exceed its maximum possible size
    std::vector<bool, limited_allocator<bool, 10> > v(5, true);
    bool a[v.max_size()] = {};
    try {
      v.insert_range(v.begin() + v.size() / 2, a);
      assert(false);
    } catch (const std::length_error&) {
      assert(v.size() == 5);
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == true);
    }
  }

  return 0;
}
