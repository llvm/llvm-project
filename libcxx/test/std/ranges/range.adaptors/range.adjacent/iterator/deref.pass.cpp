//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto operator*() const;

#include <array>
#include <cassert>
#include <cstddef>
#include <ranges>
#include <tuple>

#include "../helpers.h"
#include "../../range_adaptor_types.h"

template <std::size_t N>
constexpr void test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    // simple case
    auto v                                                      = buffer | std::views::adjacent<N>;
    std::same_as<expectedTupleType<N, int&>> decltype(auto) res = *v.begin();

    assert(&std::get<0>(res) == &buffer[0]);
    if constexpr (N >= 2)
      assert(&std::get<1>(res) == &buffer[1]);
    if constexpr (N >= 3)
      assert(&std::get<2>(res) == &buffer[2]);
    if constexpr (N >= 4)
      assert(&std::get<3>(res) == &buffer[3]);
    if constexpr (N >= 5)
      assert(&std::get<4>(res) == &buffer[4]);
  }

  {
    // operator* is const
    auto v                                                      = buffer | std::views::adjacent<N>;
    const auto it                                               = v.begin();
    std::same_as<expectedTupleType<N, int&>> decltype(auto) res = *it;
    assert(&std::get<0>(res) == &buffer[0]);
    if constexpr (N >= 2)
      assert(&std::get<1>(res) == &buffer[1]);
    if constexpr (N >= 3)
      assert(&std::get<2>(res) == &buffer[2]);
    if constexpr (N >= 4)
      assert(&std::get<3>(res) == &buffer[3]);
    if constexpr (N >= 5)
      assert(&std::get<4>(res) == &buffer[4]);
  }

  {
    // underlying range with prvalue range_reference_t
    auto v                                                     = std::views::iota(0, 8) | std::views::adjacent<N>;
    std::same_as<expectedTupleType<N, int>> decltype(auto) res = *v.begin();
    assert(std::get<0>(res) == 0);
    if constexpr (N >= 2)
      assert(std::get<1>(res) == 1);
    if constexpr (N >= 3)
      assert(std::get<2>(res) == 2);
    if constexpr (N >= 4)
      assert(std::get<3>(res) == 3);
    if constexpr (N >= 5)
      assert(std::get<4>(res) == 4);
  }

  {
    // const-correctness
    const std::array bufferConst                                      = {1, 2, 3, 4, 5, 6, 7, 8};
    auto v                                                            = bufferConst | std::views::adjacent<N>;
    std::same_as<expectedTupleType<N, const int&>> decltype(auto) res = *v.begin();
    assert(&std::get<0>(res) == &bufferConst[0]);
    if constexpr (N >= 2)
      assert(&std::get<1>(res) == &bufferConst[1]);
    if constexpr (N >= 3)
      assert(&std::get<2>(res) == &bufferConst[2]);
    if constexpr (N >= 4)
      assert(&std::get<3>(res) == &bufferConst[3]);
    if constexpr (N >= 5)
      assert(&std::get<4>(res) == &bufferConst[4]);
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
