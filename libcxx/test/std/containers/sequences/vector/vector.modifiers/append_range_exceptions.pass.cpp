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
//   constexpr void append_range(R&& rg); // C++23

#include <cassert>
#include <vector>

#include "../../insert_range_sequence_containers.h"
#include "test_allocator.h"

void test() {
  test_append_range_exception_safety_throwing_copy<std::vector>();
  test_append_range_exception_safety_throwing_allocator<std::vector, int>();

  {
    std::vector<int, limited_allocator<int, 10> > v(8, 42);
    int a[] = {1, 2, 3};
    try {
      v.append_range(a);
      assert(false);
    } catch (...) {
      assert(v.size() == 8);
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == 42);
    }
  }
}

int main(int, char**) {
  test();

  return 0;
}
