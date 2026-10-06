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

// template<container-compatible-range<bool> R>
//   constexpr void append_range(R&& rg); // C++23

#include <cassert>
#include <stdexcept>
#include <vector>

#include "../insert_range_sequence_containers.h"
#include "test_allocator.h"

int main(int, char**) {
  // Note: `test_append_range_exception_safety_throwing_copy` doesn't apply because copying booleans cannot throw.
  test_append_range_exception_safety_throwing_allocator<std::vector, bool>();

  { // Attempt to append more elements to a non-empty vector<bool> than its maximum possible size
    std::vector<bool, limited_allocator<bool, 10> > v;
    v.resize(v.max_size() - 2, true);
    bool a[] = {false, true, false};
    try {
      v.append_range(a);
      assert(false);
    } catch (const std::length_error&) {
      assert(v.size() == v.max_size() - 2);
      for (std::size_t i = 0; i < v.size(); ++i)
        assert(v[i] == true);
    }
  }
  { // Attempt to append more elements to an empty vector<bool> than its maximum possible size
    std::vector<bool, limited_allocator<bool, 10> > v;
    bool a[v.max_size() + 1] = {}; // A large enough array to trigger a length_error when appended
    try {
      v.append_range(a);
      assert(false);
    } catch (const std::length_error&) {
      assert(v.empty());
    }
  }

  return 0;
}
