//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr auto size() requires sized_range<V>
// constexpr auto size() const requires sized_range<const V>

#include <cassert>
#include <ranges>
#include <tuple>
#include <utility>

#include "types.h"

template <class T>
concept HasSize = requires(T t) { t.size(); };

static_assert(HasSize<std::ranges::elements_view<SimpleCommon, 0>>);
static_assert(HasSize<const std::ranges::elements_view<SimpleCommon, 0>>);

struct NonSized : std::ranges::view_base {
  using iterator = forward_iterator<std::tuple<int>*>;
  iterator begin() const;
  iterator end() const;
};
static_assert(!std::ranges::sized_range<NonSized>);
static_assert(!std::ranges::sized_range<const NonSized>);

static_assert(!HasSize<std::ranges::elements_view<NonSized, 0>>);
static_assert(!HasSize<const std::ranges::elements_view<NonSized, 0>>);

struct SizedNonConst : TupleBufferView {
  using TupleBufferView::TupleBufferView;

  using iterator = forward_iterator<std::tuple<int>*>;
  constexpr auto begin() const { return iterator{buffer_}; }
  constexpr auto end() const { return iterator{buffer_ + size_}; }
  constexpr std::size_t size() { return size_; }
};

static_assert(HasSize<std::ranges::elements_view<SizedNonConst, 0>>);
static_assert(!HasSize<const std::ranges::elements_view<SizedNonConst, 0>>);

struct OnlyConstSized : TupleBufferView {
  using TupleBufferView::TupleBufferView;

  using iterator = forward_iterator<std::tuple<int>*>;
  constexpr auto begin() const { return iterator{buffer_}; }
  constexpr auto end() const { return iterator{buffer_ + size_}; }
  constexpr std::size_t size() const { return size_; }
  constexpr std::size_t size() = delete;
};

static_assert(HasSize<const OnlyConstSized>);
static_assert(HasSize<std::ranges::elements_view<OnlyConstSized, 0>>);
static_assert(HasSize<const std::ranges::elements_view<OnlyConstSized, 0>>);

constexpr bool test() {
  std::tuple<int> buffer[] = {{1}, {2}, {3}};

  // non-const and const are sized
  {
    auto ev = std::views::elements<0>(buffer);
    assert(ev.size() == 3);
    assert(std::as_const(ev).size() == 3);
  }

  {
    // const-view non-sized range
    auto ev = std::views::elements<0>(SizedNonConst{buffer});
    assert(ev.size() == 3);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
