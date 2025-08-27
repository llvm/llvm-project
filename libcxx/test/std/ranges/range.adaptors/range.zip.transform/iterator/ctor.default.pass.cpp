//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// iterator() = default;

#include <ranges>

#include "../types.h"

struct IterDefaultCtrView : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

struct IterNoDefaultCtrView : std::ranges::view_base {
  cpp20_input_iterator<int*> begin() const;
  sentinel_wrapper<cpp20_input_iterator<int*>> end() const;
};

template <class... Views>
using Iter = std::ranges::iterator_t<std::ranges::zip_transform_view<MakeTuple, Views...>>;

static_assert(!std::default_initializable<Iter<IterNoDefaultCtrView>>);
static_assert(!std::default_initializable<Iter<IterNoDefaultCtrView, IterDefaultCtrView>>);
static_assert(!std::default_initializable<Iter<IterNoDefaultCtrView, IterNoDefaultCtrView>>);
static_assert(std::default_initializable<Iter<IterDefaultCtrView>>);
static_assert(std::default_initializable<Iter<IterDefaultCtrView, IterDefaultCtrView>>);

template <class Fn, class... Views>
constexpr void test() {
  using ZipTransformIter = std::ranges::iterator_t<std::ranges::zip_transform_view<Fn, Views...>>;
  ZipTransformIter iter1 = {};
  ZipTransformIter iter2;
  assert(iter1 == iter2);
}

constexpr bool test() {
  test<MakeTuple, IterDefaultCtrView>();
  test<MakeTuple, IterDefaultCtrView, std::ranges::empty_view<int>>();
  test<MakeTuple, IterDefaultCtrView, std::ranges::iota_view<int>, std::ranges::single_view<int>>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
