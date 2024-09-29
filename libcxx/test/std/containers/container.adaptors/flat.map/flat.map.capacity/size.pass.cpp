//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// size_type size() const noexcept;

#include <cassert>
#include <flat_map>
#include <vector>

#include "test_macros.h"

int main(int, char**) {
  {
    using M   = std::flat_map<int, char>;
    const M m = {{1, 'a'}, {1, 'b'}, {4, 'd'}, {5, 'e'}, {5, 'h'}};
    ASSERT_SAME_TYPE(decltype(m.size()), std::size_t);
    ASSERT_NOEXCEPT(m.size());
    assert(m.size() == 3);
  }
  {
    using M   = std::flat_map<int, char>;
    const M m = {{1, 'a'}};
    ASSERT_SAME_TYPE(decltype(m.size()), std::size_t);
    ASSERT_NOEXCEPT(m.size());
    assert(m.size() == 1);
  }
  {
    using M = std::flat_map<int, char>;
    const M m;
    ASSERT_SAME_TYPE(decltype(m.size()), std::size_t);
    ASSERT_NOEXCEPT(m.size());
    assert(m.size() == 0);
  }
  {
    using M = std::flat_map<int, char>;
    M m;
    std::size_t s = 1000000;
    for (auto i = 0u; i < s; ++i) {
      m.emplace(i, 'a');
    }
    ASSERT_SAME_TYPE(decltype(m.size()), std::size_t);
    ASSERT_NOEXCEPT(m.size());
    assert(m.size() == s);
  }

  return 0;
}
