//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr iterator& operator--() requires bidirectional_range<Base>;
// constexpr iterator operator--(int) requires bidirectional_range<Base>;

#include <array>
#include <cassert>
#include <iterator>
#include <ranges>
#include <tuple>

#include "../../range_adaptor_types.h"

template <class Iter>
concept canDecrement = requires(Iter it) { --it; } || requires(Iter it) { it--; };

template <class R, std::size_t N>
constexpr void test_one() {
  int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  const auto validateRefFromIndex = [&](auto&& tuple, std::size_t idx) {
    assert(&std::get<0>(tuple) == &buffer[idx]);
    if constexpr (N >= 2)
      assert(&std::get<1>(tuple) == &buffer[idx + 1]);
    if constexpr (N >= 3)
      assert(&std::get<2>(tuple) == &buffer[idx + 2]);
    if constexpr (N >= 4)
      assert(&std::get<3>(tuple) == &buffer[idx + 3]);
    if constexpr (N >= 5)
      assert(&std::get<4>(tuple) == &buffer[idx + 4]);
  };

  {
    auto v = R(buffer) | std::views::adjacent<N>;

    auto it    = v.begin();
    using Iter = decltype(it);

    std::ranges::advance(it, v.end());

    --it;
    validateRefFromIndex(*it, 9 - N);

    static_assert(std::is_same_v<decltype(--it), Iter&>);
    std::same_as<Iter&> decltype(auto) it_ref = --it;
    assert(&it_ref == &it);

    validateRefFromIndex(*it, 8 - N);

    std::same_as<Iter> decltype(auto) tmp = it--;

    validateRefFromIndex(*tmp, 8 - N);
    validateRefFromIndex(*it, 7 - N);

    // Decrement to the beginning
    for (int i = 6 - N; i >= 0; --i) {
      --it;
      validateRefFromIndex(*it, i);
    }
    assert(it == v.begin());
  }
}

template <std::size_t N>
constexpr void test() {
  test_one<ContiguousNonCommonSized, N>();
  test_one<SimpleCommonRandomAccessSized, N>();
  test_one<BidiNonCommonView, N>();

  // Non-bidirectional base range
  {
    using View = std::ranges::adjacent_view<ForwardSizedView, N>;
    using Iter = std::ranges::iterator_t<View>;

    static_assert(!canDecrement<Iter>);
  }
}

constexpr bool test() {
  test<1>();
  test<2>();
  test<3>();
  test<4>();
  test<5>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
