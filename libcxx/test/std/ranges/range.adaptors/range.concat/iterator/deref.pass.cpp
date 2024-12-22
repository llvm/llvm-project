//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <utility>
#include "test_iterators.h"
#include "test_macros.h"
#include "../types.h"

template <class Iter, class ValueType = int, class Sent = sentinel_wrapper<Iter>>
constexpr void test() {
  using View = minimal_view<Iter, Sent>;
  using ConcatView = std::ranges::concat_view<View>;
  using ConcatIterator = std::ranges::iterator_t<ConcatView>;

  auto make_concat_view = [](auto begin, auto end) {
    View view{Iter(begin), Sent(Iter(end))};
    return ConcatView(std::move(view));
  };

  std::array array{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  ConcatView view = make_concat_view(array.data(), array.data() + array.size());
  ConcatIterator iter = view.begin(); 
  int& result = *iter; 
  ASSERT_SAME_TYPE(int&, decltype(*iter));
  assert(&result == array.data()); 

  
}

constexpr bool tests() {
  test<cpp17_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
