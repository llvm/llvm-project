//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// bool contains(const key_type& x) const;

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
    assert(!m.contains(0));
    assert(m.contains(1));
    assert(m.contains(2));
    assert(!m.contains(3));
    assert(m.contains(4));
    assert(m.contains(5));
    assert(!m.contains(6));
    assert(!m.contains(7));
    assert(std::as_const(m).contains(8));
    assert(!std::as_const(m).contains(9));
    m.clear();
    assert(!m.contains(1));
  }
  {
    using M = std::flat_map<int, int, std::greater<int>, std::deque<int, min_allocator<int>>>;
    M m     = {{1, 0}, {2, 0}, {4, 0}, {5, 0}, {8, 0}};
    assert(!m.contains(0));
    assert(m.contains(1));
    assert(m.contains(2));
    assert(!m.contains(3));
    assert(m.contains(4));
    assert(m.contains(5));
    assert(!m.contains(6));
    assert(!m.contains(7));
    assert(std::as_const(m).contains(8));
    assert(!std::as_const(m).contains(9));
    m.clear();
    assert(!m.contains(1));
  }
  {
    using M = std::flat_map<bool, int>;
    M m     = {{true, 1}, {false, 2}};
    assert(m.contains(true));
    assert(m.contains(false));
    m = {{true, 3}};
    assert(m.contains(true));
    assert(!m.contains(false));
    m = {{false, 4}};
    assert(!std::as_const(m).contains(true));
    assert(std::as_const(m).contains(false));
    m.clear();
    assert(!m.contains(true));
    assert(!m.contains(false));
  }
  return 0;
}
