//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// class flat_multimap

// [[nodiscard]] bool empty() const noexcept;

#include <flat_map>
#include <cassert>
#include <deque>
#include <functional>
#include <utility>
#include <vector>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class KeyContainer, class ValueContainer>
void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_multimap<Key, Value, std::less<int>, KeyContainer, ValueContainer>;
  M m;
  ASSERT_SAME_TYPE(decltype(m.empty()), bool);
  ASSERT_NOEXCEPT(m.empty());
  assert(m.empty());
  assert(std::as_const(m).empty());
  m = {{1, 1.0}, {1, 2.0}};
  assert(!m.empty());
  m.clear();
  assert(m.empty());
}

int main(int, char**) {
  test<std::vector<int>, std::vector<double>>();
  test<std::deque<int>, std::vector<double>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<double>>();
  test<std::vector<int, min_allocator<int>>, std::vector<double, min_allocator<double>>>();

  return 0;
}
