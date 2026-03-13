//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// [[nodiscard]] bool empty() const noexcept;

#include <flat_set>
#include <cassert>
#include <deque>
#include <functional>
#include <utility>
#include <vector>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class KeyContainer>
constexpr void test_one() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_set<Key, std::less<int>, KeyContainer>;
  M m;
  ASSERT_SAME_TYPE(decltype(m.empty()), bool);
  ASSERT_NOEXCEPT(m.empty());
  assert(m.empty());
  assert(std::as_const(m).empty());
  m = {1};
  assert(!m.empty());
  m.clear();
  assert(m.empty());
}

constexpr bool test() {
  test_one<std::vector<int>>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
    test_one<std::deque<int>>();
  test_one<MinSequenceContainer<int>>();
  test_one<std::vector<int, min_allocator<int>>>();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
