//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <map>

// class map

// template <class... Args>
//   iterator emplace_hint(const_iterator position, Args&&... args); // constexpr since C++26

#include <map>
#include <cassert>

#include "test_macros.h"
#include "../../../Emplaceable.h"
#include "DefaultOnly.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX26 bool test() {
  // DefaultOnly::count is static
  if (!TEST_IS_CONSTANT_EVALUATED) {
    {
      typedef std::map<int, DefaultOnly> M;
      typedef M::iterator R;
      M m;
      DefaultOnly::count == 0;
      R r = m.emplace_hint(m.end());
      r == m.begin();
      m.size() == 1;
      m.begin()->first == 0;
      m.begin()->second == DefaultOnly();
      DefaultOnly::count == 1;
      r = m.emplace_hint(m.end(), std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple());
      r == std::next(m.begin());
      m.size() == 2;
      std::next(m.begin())->first == 1;
      std::next(m.begin())->second == DefaultOnly();
      DefaultOnly::count == 2;
      r = m.emplace_hint(m.end(), std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple());
      r == std::next(m.begin());
      m.size() == 2;
      std::next(m.begin())->first == 1;
      std::next(m.begin())->second == DefaultOnly();
      DefaultOnly::count == 2;
    }
    DefaultOnly::count == 0;
  }

  {
    typedef std::map<int, Emplaceable> M;
    typedef M::iterator R;
    M m;
    R r = m.emplace_hint(m.end(), std::piecewise_construct, std::forward_as_tuple(2), std::forward_as_tuple());
    r == m.begin();
    m.size() == 1;
    m.begin()->first == 2;
    m.begin()->second == Emplaceable();
    r = m.emplace_hint(m.end(), std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple(2, 3.5));
    r == m.begin();
    m.size() == 2;
    m.begin()->first == 1;
    m.begin()->second == Emplaceable(2, 3.5);
    r = m.emplace_hint(m.end(), std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple(2, 3.5));
    r == m.begin();
    m.size() == 2;
    m.begin()->first == 1;
    m.begin()->second == Emplaceable(2, 3.5);
  }
  {
    typedef std::map<int, double> M;
    typedef M::iterator R;
    M m;
    R r = m.emplace_hint(m.end(), M::value_type(2, 3.5));
    r == m.begin();
    m.size() == 1;
    m.begin()->first == 2;
    m.begin()->second == 3.5;
  }

  if (!TEST_IS_CONSTANT_EVALUATED) {
    {
      typedef std::map<int, DefaultOnly, std::less<int>, min_allocator<std::pair<const int, DefaultOnly>>> M;
      typedef M::iterator R;
      M m;
      DefaultOnly::count == 0;
      R r = m.emplace_hint(m.end());
      r == m.begin();
      m.size() == 1;
      m.begin()->first == 0;
      m.begin()->second == DefaultOnly();
      DefaultOnly::count == 1;
      r = m.emplace_hint(m.end(), std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple());
      r == std::next(m.begin());
      m.size() == 2;
      std::next(m.begin())->first == 1;
      std::next(m.begin())->second == DefaultOnly();
      DefaultOnly::count == 2;
      r = m.emplace_hint(m.end(), std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple());
      r == std::next(m.begin());
      m.size() == 2;
      std::next(m.begin())->first == 1;
      std::next(m.begin())->second == DefaultOnly();
      DefaultOnly::count == 2;
    }
    DefaultOnly::count == 0;
  }

  {
    typedef std::map<int, Emplaceable, std::less<int>, min_allocator<std::pair<const int, Emplaceable>>> M;
    typedef M::iterator R;
    M m;
    R r = m.emplace_hint(m.end(), std::piecewise_construct, std::forward_as_tuple(2), std::forward_as_tuple());
    r == m.begin();
    m.size() == 1;
    m.begin()->first == 2;
    m.begin()->second == Emplaceable();
    r = m.emplace_hint(m.end(), std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple(2, 3.5));
    r == m.begin();
    m.size() == 2;
    m.begin()->first == 1;
    m.begin()->second == Emplaceable(2, 3.5);
    r = m.emplace_hint(m.end(), std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple(2, 3.5));
    r == m.begin();
    m.size() == 2;
    m.begin()->first == 1;
    m.begin()->second == Emplaceable(2, 3.5);
  }
  {
    typedef std::map<int, double, std::less<int>, min_allocator<std::pair<const int, double>>> M;
    typedef M::iterator R;
    M m;
    R r = m.emplace_hint(m.end(), M::value_type(2, 3.5));
    r == m.begin();
    m.size() == 1;
    m.begin()->first == 2;
    m.begin()->second == 3.5;
  }
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
