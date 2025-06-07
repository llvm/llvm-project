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
#include <vector>

#include "../insert_range_sequence_containers.h"
#include "test_allocator.h"

void test() {
  // Note: `test_append_range_exception_safety_throwing_copy` doesn't apply because copying booleans cannot throw.
  test_append_range_exception_safety_throwing_allocator<std::vector, bool>();

  {
    using Vec = std::vector<bool, limited_allocator<bool, 10> >;
    Vec v(Vec().max_size() - 2, true);
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
  {
    std::vector<bool, limited_allocator<bool, 10> > v(10, true);
    bool a[10 * 65536] = {};
    try {
      v.append_range(a);
      assert(false);
    } catch (...) {
      assert(v.size() == 10);
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == true);
    }
  }
}

int main(int, char**) {
  test();

  return 0;
}
