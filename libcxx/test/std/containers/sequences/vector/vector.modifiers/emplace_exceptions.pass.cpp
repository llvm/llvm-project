//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: no-exceptions

// <vector>

// template <class... Args> iterator emplace(const_iterator pos, Args&&... args);

#include <cassert>
#include <stdexcept>
#include <vector>

#include "test_allocator.h"

int main(int, char**) {
  { // Attempt to emplace an element at the beginning of a vector that is already at its maximum possible size
    std::vector<int, limited_allocator<int, 10> > v(10, 42);
    try {
      v.emplace(v.begin(), 0);
      assert(false);
    } catch (const std::length_error&) {
      assert(v.size() == v.max_size());
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == 42);
    }
  }
  { // Attempt to emplace an element at the end of a vector that is already at its maximum possible size
    std::vector<int, limited_allocator<int, 10> > v(10, 42);
    try {
      v.emplace(v.end(), 0);
      assert(false);
    } catch (const std::length_error&) {
      assert(v.size() == v.max_size());
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == 42);
    }
  }
  { // Attempt to emplace an element in the middle of a vector that is already at its maximum possible size
    std::vector<int, limited_allocator<int, 10> > v(10, 42);
    try {
      v.emplace(v.begin() + v.size() / 2, 0);
      assert(false);
    } catch (const std::length_error&) {
      assert(v.size() == v.max_size());
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == 42);
    }
  }

  return 0;
}
