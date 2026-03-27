//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr iterator& operator++();
// constexpr iterator operator++(int);

#include <array>
#include <cassert>
#include <concepts>
#include <ranges>
#include <tuple>

#include "../../range_adaptor_types.h"

template <class R, std::size_t N>
constexpr void test() {
  int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8};

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
    auto v     = R(buffer) | std::views::adjacent<N>;
    auto it    = v.begin();
    using Iter = decltype(it);

    validateRefFromIndex(*it, 0);

    std::same_as<Iter&> decltype(auto) it_ref = ++it;
    assert(&it_ref == &it);

    validateRefFromIndex(*it, 1);

    static_assert(std::is_same_v<decltype(it++), Iter>);
    auto original                          = it;
    std::same_as<Iter> decltype(auto) copy = it++;
    assert(original == copy);

    validateRefFromIndex(*copy, 1);
    validateRefFromIndex(*it, 2);

    // Increment to the end
    for (std::size_t i = 3; i != 9 - N; ++i) {
      ++it;
      validateRefFromIndex(*it, i);
    }

    ++it;
    assert(it == v.end());
  }
}

template <std::size_t N>
constexpr void test() {
  test<ContiguousCommonView, N>();
  test<SimpleCommonRandomAccessSized, N>();
  test<BidiNonCommonView, N>();
  test<ForwardSizedView, N>();
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
