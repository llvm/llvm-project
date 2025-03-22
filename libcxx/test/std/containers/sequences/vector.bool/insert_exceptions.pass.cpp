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
    using Vec = std::vector<bool, limited_allocator<bool, 10> >;
    Vec v(Vec().max_size(), true);
    try {
      v.insert(v.begin(), false);
      assert(false);
    } catch (...) {
      assert(v.size() == v.max_size());
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == true);
    }
  }
  {
    using Vec = std::vector<bool, limited_allocator<bool, 10> >;
    Vec v(Vec().max_size(), false);
    try {
      v.insert(v.end(), 0);
      assert(false);
    } catch (...) {
      assert(v.size() == v.max_size());
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == false);
    }
  }
  {
    using Vec = std::vector<bool, limited_allocator<bool, 10> >;
    Vec v(Vec().max_size(), true);
    try {
      v.insert(v.begin() + v.size() / 2, false);
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
      v.insert(v.begin(), 1, false);
      assert(false);
    } catch (...) {
      assert(v.size() == v.max_size());
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == true);
    }
  }
  {
    using Vec = std::vector<bool, limited_allocator<bool, 10> >;
    Vec v(Vec().max_size() - 2, true);
    bool a[] = {true, false, true};
    try {
      v.insert(v.begin() + v.size() / 2, std::begin(a), std::end(a));
      assert(false);
    } catch (...) {
      assert(v.size() == v.max_size() - 2);
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == true);
    }
  }
#if TEST_STD_VER >= 11
  {
    using Vec = std::vector<bool, limited_allocator<bool, 10> >;
    Vec v(Vec().max_size() - 2, true);
    try {
      v.insert(v.begin(), {true, false, true});
      assert(false);
    } catch (...) {
      assert(v.size() == v.max_size() - 2);
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == true);
    }
  }
#endif
}

int main(int, char**) {
  test();

  return 0;
}
