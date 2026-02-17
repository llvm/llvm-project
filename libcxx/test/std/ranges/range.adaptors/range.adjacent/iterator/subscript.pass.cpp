//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto operator[](difference_type n) const requires
//        all_random_access<Const, Views...>

#include <ranges>
#include <cassert>

#include "../helpers.h"
#include "../../range_adaptor_types.h"

template <std::size_t N>
constexpr void test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    // random_access_range
    std::ranges::adjacent_view<SizedRandomAccessView, N> v(SizedRandomAccessView{buffer});
    auto it = v.begin();
    assert(it[0] == *it);
    assert(it[2] == *(it + 2));
    assert(&std::get<0>(it[2]) == &buffer[2]);
    if constexpr (N >= 2)
      assert(&std::get<1>(it[2]) == &buffer[3]);
    if constexpr (N >= 3)
      assert(&std::get<2>(it[2]) == &buffer[4]);
    if constexpr (N >= 4)
      assert(&std::get<3>(it[2]) == &buffer[5]);
    if constexpr (N >= 5)
      assert(&std::get<4>(it[2]) == &buffer[6]);

    static_assert(std::is_same_v<decltype(it[0]), expectedTupleType<N, int&>>);
  }

  {
    // contiguous_range
    std::ranges::adjacent_view<ContiguousCommonView, N> v(ContiguousCommonView{buffer});
    auto it = v.begin();
    assert(it[0] == *it);
    assert(it[2] == *(it + 2));
    assert(&std::get<0>(it[0]) == &buffer[0]);
    if constexpr (N >= 2)
      assert(&std::get<1>(it[0]) == &buffer[1]);
    if constexpr (N >= 3)
      assert(&std::get<2>(it[0]) == &buffer[2]);
    if constexpr (N >= 4)
      assert(&std::get<3>(it[0]) == &buffer[3]);
    if constexpr (N >= 5)
      assert(&std::get<4>(it[0]) == &buffer[4]);

    static_assert(std::is_same_v<decltype(it[0]), expectedTupleType<N, int&>>);
  }

  {
    // underlying range with prvalue range_reference_t
    auto v                                                     = std::views::iota(0, 8) | std::views::adjacent<N>;
    std::same_as<expectedTupleType<N, int>> decltype(auto) res = v.begin()[2];
    assert(std::get<0>(res) == 2);
    if constexpr (N >= 2)
      assert(std::get<1>(res) == 3);
    if constexpr (N >= 3)
      assert(std::get<2>(res) == 4);
    if constexpr (N >= 4)
      assert(std::get<3>(res) == 5);
    if constexpr (N >= 5)
      assert(std::get<4>(res) == 6);
  }

  {
    // non random_access_range
    std::ranges::adjacent_view<BidiCommonView, N> v(BidiCommonView{buffer});
    auto iter               = v.begin();
    const auto canSubscript = [](auto&& it) { return requires { it[0]; }; };
    static_assert(!canSubscript(iter));
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
