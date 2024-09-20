//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// size_type count(const key_type& x) const;

#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
#include <utility>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    using M = std::flat_map<int, const char*>;
    M m     = {{1, ""}, {2, ""}, {4, ""}, {5, ""}, {8, ""}};
    ASSERT_SAME_TYPE(decltype(m.count(0)), size_t);
    assert(m.count(0) == 0);
    assert(m.count(1) == 1);
    assert(m.count(2) == 1);
    assert(m.count(3) == 0);
    assert(m.count(4) == 1);
    assert(m.count(5) == 1);
    assert(m.count(6) == 0);
    assert(m.count(7) == 0);
    assert(std::as_const(m).count(8) == 1);
    assert(std::as_const(m).count(9) == 0);
  }
  {
    using M = std::flat_map<int, int, std::greater<int>, std::deque<int, min_allocator<int>>>;
    M m     = {{1, 0}, {2, 0}, {4, 0}, {5, 0}, {8, 0}};
    ASSERT_SAME_TYPE(decltype(m.count(0)), size_t);
    assert(m.count(0) == 0);
    assert(m.count(1) == 1);
    assert(m.count(2) == 1);
    assert(m.count(3) == 0);
    assert(m.count(4) == 1);
    assert(m.count(5) == 1);
    assert(m.count(6) == 0);
    assert(m.count(7) == 0);
    assert(std::as_const(m).count(8) == 1);
    assert(std::as_const(m).count(9) == 0);
  }
  {
    using M = std::flat_map<bool, int>;
    M m     = {{true, 1}, {false, 2}};
    ASSERT_SAME_TYPE(decltype(m.count(0)), size_t);
    assert(m.count(true) == 1);
    assert(m.count(false) == 1);
    m = {{true, 3}};
    assert(m.count(true) == 1);
    assert(m.count(false) == 0);
    m = {{false, 4}};
    assert(std::as_const(m).count(true) == 0);
    assert(std::as_const(m).count(false) == 1);
    m.clear();
    assert(m.count(true) == 0);
    assert(m.count(false) == 0);
  }
  return 0;
}
