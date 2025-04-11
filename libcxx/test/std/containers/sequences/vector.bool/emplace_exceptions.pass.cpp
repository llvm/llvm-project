//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: no-exceptions

// <vector>
// vector<bool>

// template <class... Args> iterator emplace(const_iterator pos, Args&&... args);

#include <cassert>
#include <vector>

#include "test_allocator.h"

void test() {
  {
    using Vec = std::vector<bool, limited_allocator<bool, 10> >;
    Vec v(Vec().max_size(), true);
    try {
      v.emplace(v.begin(), true);
      assert(false);
    } catch (...) {
      assert(v.size() == v.max_size());
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == true);
    }
  }
  {
    using Vec = std::vector<bool, limited_allocator<bool, 10> >;
    Vec v(Vec().max_size(), true);
    try {
      v.emplace(v.end(), true);
      assert(false);
    } catch (...) {
      assert(v.size() == v.max_size());
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == true);
    }
  }
  {
    using Vec = std::vector<bool, limited_allocator<bool, 10> >;
    Vec v(Vec().max_size(), true);
    try {
      v.emplace(v.begin() + v.size() / 2, true);
      assert(false);
    } catch (...) {
      assert(v.size() == v.max_size());
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == true);
    }
  }
}

int main(int, char**) {
  test();

  return 0;
}
