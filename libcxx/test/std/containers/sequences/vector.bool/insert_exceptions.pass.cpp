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
#include <stdexcept>
#include <vector>

#include "test_allocator.h"
#include "test_macros.h"

int main(int, char**) {
  { // Attempt to insert an element at the beginning of a vector<bool> that is already at its maximum possible size
    std::vector<bool, limited_allocator<bool, 10> > v;
    v.resize(v.max_size(), true);
    try {
      v.insert(v.begin(), false);
      assert(false);
    } catch (const std::length_error&) {
      assert(v.size() == v.max_size());
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == true);
    }
  }
  { // Attempt to insert an element at the end of a vector<bool> that is already at its maximum possible size
    std::vector<bool, limited_allocator<bool, 10> > v;
    v.resize(v.max_size(), false);
    try {
      v.insert(v.end(), 0);
      assert(false);
    } catch (const std::length_error&) {
      assert(v.size() == v.max_size());
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == false);
    }
  }
  { // Attempt to insert an element in the middle of a vector<bool> that is already at its maximum possible size
    std::vector<bool, limited_allocator<bool, 10> > v;
    v.resize(v.max_size(), true);
    try {
      v.insert(v.begin() + v.size() / 2, false);
      assert(false);
    } catch (const std::length_error&) {
      assert(v.size() == v.max_size());
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == true);
    }
  }
  { // Attempt to insert an iterator range to a vector<bool> that would exceed its maximum possible size
    std::vector<bool, limited_allocator<bool, 10> > v;
    v.resize(v.max_size() - 2, true);
    bool a[] = {true, false, true};
    try {
      v.insert(v.begin() + v.size() / 2, std::begin(a), std::end(a));
      assert(false);
    } catch (const std::length_error&) {
      assert(v.size() == v.max_size() - 2);
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == true);
    }
  }
#if TEST_STD_VER >= 11
  { // Attempt to insert an initializer_list to a vector<bool> that would exceed its maximum possible size
    std::vector<bool, limited_allocator<bool, 10> > v;
    v.resize(v.max_size() - 2, true);
    try {
      v.insert(v.begin(), {true, false, true});
      assert(false);
    } catch (const std::length_error&) {
      assert(v.size() == v.max_size() - 2);
      for (std::size_t i = 0; i != v.size(); ++i)
        assert(v[i] == true);
    }
  }
#endif

  return 0;
}
