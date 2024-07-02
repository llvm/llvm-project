//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// There is a bug in older versions of Clang that causes trouble with constraints in classes like
// `ContainerWithDirectCtr`.
// XFAIL: apple-clang-15

// template<template<class...> class C, input_range R, class... Args>
//   constexpr auto to(R&& r, Args&&... args);  // Since C++23

#include <ranges>

#include <algorithm>
#include <array>
#include <cassert>
#include "container.h"

template <class ElementType>
struct ContainerWithDirectCtr : Container<ElementType, CtrChoice::DirectCtr> {
  using Container<ElementType, CtrChoice::DirectCtr>::Container;
};

template <std::ranges::input_range Range>
ContainerWithDirectCtr(Range&&) -> ContainerWithDirectCtr<std::ranges::range_value_t<Range>>;

template <std::ranges::input_range Range>
ContainerWithDirectCtr(Range&&, int, char) -> ContainerWithDirectCtr<std::ranges::range_value_t<Range>>;

template <class ElementType>
struct ContainerWithFromRangeT : Container<ElementType, CtrChoice::FromRangeT> {
  using Container<ElementType, CtrChoice::FromRangeT>::Container;
};

template <std::ranges::input_range Range>
ContainerWithFromRangeT(std::from_range_t, Range&&) -> ContainerWithFromRangeT<std::ranges::range_value_t<Range>>;

template <std::ranges::input_range Range>
ContainerWithFromRangeT(std::from_range_t, Range&&, int, char) ->
    ContainerWithFromRangeT<std::ranges::range_value_t<Range>>;

template <class ElementType>
struct ContainerWithBeginEndPair : Container<ElementType, CtrChoice::BeginEndPair> {
  using Container<ElementType, CtrChoice::BeginEndPair>::Container;
};

template <class Iter>
ContainerWithBeginEndPair(Iter, Iter) -> ContainerWithBeginEndPair<std::iter_value_t<Iter>>;

template <class Iter>
ContainerWithBeginEndPair(Iter, Iter, int, char) -> ContainerWithBeginEndPair<std::iter_value_t<Iter>>;

constexpr bool test() {
  std::array in = {1, 2, 3, 4, 5};
  int arg1 = 42;
  char arg2 = 'a';

  { // Case 1 -- can construct directly from the given range.
    {
      std::same_as<ContainerWithDirectCtr<int>> decltype(auto) result = std::ranges::to<ContainerWithDirectCtr>(in);

      assert(result.ctr_choice == CtrChoice::DirectCtr);
      assert(std::ranges::equal(result, in));
      assert((in | std::ranges::to<ContainerWithDirectCtr>()) == result);
    }

    { // Extra arguments.
      std::same_as<ContainerWithDirectCtr<int>> decltype(auto) result =
          std::ranges::to<ContainerWithDirectCtr>(in, arg1, arg2);

      assert(result.ctr_choice == CtrChoice::DirectCtr);
      assert(std::ranges::equal(result, in));
      assert(result.extra_arg1 == arg1);
      assert(result.extra_arg2 == arg2);
      assert((in | std::ranges::to<ContainerWithDirectCtr>(arg1, arg2)) == result);
    }
  }

  { // Case 2 -- can construct from the given range using the `from_range_t` tagged constructor.
    {
      std::same_as<ContainerWithFromRangeT<int>> decltype(auto) result = std::ranges::to<ContainerWithFromRangeT>(in);

      assert(result.ctr_choice == CtrChoice::FromRangeT);
      assert(std::ranges::equal(result, in));
      assert((in | std::ranges::to<ContainerWithFromRangeT>()) == result);
    }

    { // Extra arguments.
      std::same_as<ContainerWithFromRangeT<int>> decltype(auto) result =
          std::ranges::to<ContainerWithFromRangeT>(in, arg1, arg2);

      assert(result.ctr_choice == CtrChoice::FromRangeT);
      assert(std::ranges::equal(result, in));
      assert(result.extra_arg1 == arg1);
      assert(result.extra_arg2 == arg2);
      assert((in | std::ranges::to<ContainerWithFromRangeT>(arg1, arg2)) == result);
    }
  }

  { // Case 3 -- can construct from a begin-end iterator pair.
    {
      std::same_as<ContainerWithBeginEndPair<int>> decltype(auto) result =
          std::ranges::to<ContainerWithBeginEndPair>(in);

      assert(result.ctr_choice == CtrChoice::BeginEndPair);
      assert(std::ranges::equal(result, in));
      assert((in | std::ranges::to<ContainerWithBeginEndPair>()) == result);
    }

    { // Extra arguments.
      std::same_as<ContainerWithBeginEndPair<int>> decltype(auto) result =
          std::ranges::to<ContainerWithBeginEndPair>(in, arg1, arg2);

      assert(result.ctr_choice == CtrChoice::BeginEndPair);
      assert(std::ranges::equal(result, in));
      assert(result.extra_arg1 == arg1);
      assert(result.extra_arg2 == arg2);
      assert((in | std::ranges::to<ContainerWithBeginEndPair>(arg1, arg2)) == result);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
