//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// [[nodiscard]] bool empty() const noexcept;

#include <flat_map>
#include <cassert>
#include <deque>
#include <functional>
#include <utility>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    typedef std::flat_map<int, int> M;
    M m;
    ASSERT_SAME_TYPE(decltype(m.empty()), bool);
    ASSERT_NOEXCEPT(m.empty());
    assert(m.empty());
    assert(std::as_const(m).empty());
    m = {{1, 1}};
    assert(!m.empty());
    m.clear();
    assert(m.empty());
  }
  {
    typedef std::flat_map<int, int, std::less<int>, std::deque<int, min_allocator<int>>> M;
    M m;
    ASSERT_SAME_TYPE(decltype(m.empty()), bool);
    ASSERT_NOEXCEPT(m.empty());
    assert(m.empty());
    assert(std::as_const(m).empty());
    m = {{1, 1}};
    assert(!m.empty());
    m.clear();
    assert(m.empty());
  }
#if 0
  // vector<bool> is not supported
  {
    typedef std::flat_map<bool, bool> M;
    M m;
    ASSERT_SAME_TYPE(decltype(m.empty()), bool);
    ASSERT_NOEXCEPT(m.empty());
    assert(m.empty());
    assert(std::as_const(m).empty());
    m = {{false, false}};
    assert(!m.empty());
    m.clear();
    assert(m.empty());
  }
#endif
  return 0;
}
