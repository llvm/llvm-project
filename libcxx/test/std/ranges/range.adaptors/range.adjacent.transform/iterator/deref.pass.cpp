//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr decltype(auto) operator*() const noexcept(see below);

#include <array>
#include <cassert>
#include <cstddef>
#include <numeric>
#include <ranges>
#include <tuple>

#include "../helpers.h"
#include "../../range_adaptor_types.h"

template <class Iter>
concept DerefNoexcept = requires(const Iter iter) {
  { *iter } noexcept;
};

static_assert(DerefNoexcept<int*>);

struct NoExceptInvocable {
  int operator()(auto...) const noexcept;
};

struct ThrowingInvocable {
  int operator()(auto...) const;
};

template <std::size_t N>
constexpr void test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Remarks: Let Is be the pack 0, 1, ..., (N - 1). The exception specification is equivalent to:
  // noexcept(invoke(*parent_->fun_, *std::get<Is>(inner_.current_)...))
  {
    using View = std::ranges::adjacent_transform_view<SimpleCommon, NoExceptInvocable, N>;
    static_assert(DerefNoexcept<std::ranges::iterator_t<View>>);
  }
  {
    using View = std::ranges::adjacent_transform_view<SimpleCommon, ThrowingInvocable, N>;
    static_assert(!DerefNoexcept<std::ranges::iterator_t<View>>);
  }
  {
    // underlying iter deref not noexcept
    static_assert(!DerefNoexcept<std::ranges::iterator_t<ForwardSizedView>>);
    using View = std::ranges::adjacent_transform_view<ForwardSizedView, NoExceptInvocable, N>;
    static_assert(!DerefNoexcept<std::ranges::iterator_t<View>>);
  }

  {
    // make_tuple
    auto v = buffer | std::views::adjacent_transform<N>(MakeTuple{});
    std::same_as<expectedTupleType<N, int>> decltype(auto) res = *v.begin();
    ValidateTupleFromIndex<N>{}(buffer, res, 0);
  }

  {
    // tie
    auto v                                                      = buffer | std::views::adjacent_transform<N>(Tie{});
    std::same_as<expectedTupleType<N, int&>> decltype(auto) res = *v.begin();
    ValidateTieFromIndex<N>{}(buffer, res, 0);
  }

  {
    // get<0>
    auto v                                = buffer | std::views::adjacent_transform<N>(GetFirst{});
    std::same_as<int&> decltype(auto) res = *v.begin();
    assert(&res == &buffer[0]);
  }

  {
    // Multiply
    auto v                               = buffer | std::views::adjacent_transform<N>(Multiply{});
    std::same_as<int> decltype(auto) res = *v.begin();
    auto expected                        = std::accumulate(buffer, buffer + N, 1, std::multiplies<>());
    assert(res == expected);
  }

  {
    // operator* is const
    auto v                                = buffer | std::views::adjacent_transform<N>(GetFirst{});
    const auto it                         = v.begin();
    std::same_as<int&> decltype(auto) res = *it;
    assert(&res == &buffer[0]);
  }

  {
    // underlying range with prvalue range_reference_t
    auto v = std::views::iota(0, 8) | std::views::adjacent_transform<N>(MakeTuple{});
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
    const std::array bufferConst = {1, 2, 3, 4, 5, 6, 7, 8};
    auto v                       = bufferConst | std::views::adjacent_transform<N>(Tie{});
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
