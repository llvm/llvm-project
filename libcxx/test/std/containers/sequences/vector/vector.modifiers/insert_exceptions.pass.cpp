//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// <vector>

// iterator insert(const_iterator position, const value_type& x);
// iterator insert(const_iterator position, size_type n, const value_type& x);
// iterator insert(const_iterator p, initializer_list<value_type> il);
// template <class Iter>
//   iterator insert(const_iterator position, Iter first, Iter last);

#include <cassert>
#include <vector>

#include "test_allocator.h"
#include "test_macros.h"

void test() {
  {
    std::vector<int, limited_allocator<int, 10> > v(10, 42);
    try {
      v.insert(v.begin(), 0);
      assert(false);
    } catch (...) {
      assert(v.size() == v.max_size());
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == 42);
    }
  }
  {
    std::vector<int, limited_allocator<int, 10> > v(10, 42);
    try {
      v.insert(v.end(), 0);
      assert(false);
    } catch (...) {
      assert(v.size() == v.max_size());
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == 42);
    }
  }
  {
    std::vector<int, limited_allocator<int, 10> > v(10, 42);
    try {
      v.insert(v.begin() + v.size() / 2, 0);
      assert(false);
    } catch (...) {
      assert(v.size() == v.max_size());
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == 42);
    }
  }
  {
    std::vector<int, limited_allocator<int, 10> > v(8, 42);
    try {
      v.insert(v.begin(), 3, 0);
      assert(false);
    } catch (...) {
      assert(v.size() == 8);
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == 42);
    }
  }
  {
    std::vector<int, limited_allocator<int, 10> > v(8, 42);
    int a[] = {1, 2, 3};
    try {
      v.insert(v.begin() + v.size() / 2, std::begin(a), std::end(a));
      assert(false);
    } catch (...) {
      assert(v.size() == 8);
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == 42);
    }
  }
#if TEST_STD_VER >= 11
  {
    std::vector<int, limited_allocator<int, 10> > v(8, 42);
    try {
      v.insert(v.begin(), {1, 2, 3});
      assert(false);
    } catch (...) {
      assert(v.size() == 8);
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == 42);
    }
  }
#endif
}

int main(int, char**) {
  test();

  return 0;
}
