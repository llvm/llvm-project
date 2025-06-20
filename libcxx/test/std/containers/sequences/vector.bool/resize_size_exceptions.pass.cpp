//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// <vector>
// vector<bool>

// void resize(size_type sz);

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <vector>

#include "test_allocator.h"

int main(int, char**) {
  {
    std::vector<bool, limited_allocator<bool, 10> > v(5, false);
    try {
      v.resize(v.max_size() + 1);
      assert(false);
    } catch (const std::length_error&) {
      assert(v.size() == 5);
      assert(v.capacity() >= 5);
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(!v[i]);
    }
  }
  {
    std::vector<bool, limited_allocator<bool, 10> > v;
    v.resize(v.max_size() / 2);
    try {
      v.resize(v.max_size() + 1);
      assert(false);
    } catch (const std::length_error&) {
      assert(v.size() == v.max_size() / 2);
      for (std::size_t i = 0; i < v.size(); ++i)
        assert(v[i] == false);
    }
  }
  {
    bool a[] = {true, false, true, false, true};
    std::vector<bool> v(std::begin(a), std::end(a));
    try {
      v.resize(v.max_size() + 1);
      assert(false);
    } catch (const std::length_error&) {
      assert(v.size() == 5);
      assert(v.capacity() >= 5);
      assert(std::equal(v.begin(), v.end(), std::begin(a)));
    }
  }

  return 0;
}
