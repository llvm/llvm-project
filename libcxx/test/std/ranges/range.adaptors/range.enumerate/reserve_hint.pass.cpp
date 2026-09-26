//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

// class enumerate_view

// constexpr auto reserve_hint()
//     requires approximately_sized_range<V>;
// constexpr auto reserve_hint() const
//     requires approximately_sized_range<const V>;

#include <cassert>
#include <ranges>
#include <utility>

#include "test_iterators.h"
#include "types.h"

template <class T>
concept HasMemberReserveHint = requires(T t) { t.reserve_hint(); };

static constexpr int globalBuff[8] = {};

struct NonApproximatelySizedView : std::ranges::view_base {
  using iterator = forward_iterator<const int*>;
  iterator begin() const;
  iterator end() const;
};

static_assert(!std::ranges::approximately_sized_range<NonApproximatelySizedView>);
static_assert(!std::ranges::approximately_sized_range<const NonApproximatelySizedView>);

static_assert(!HasMemberReserveHint<std::ranges::enumerate_view<NonApproximatelySizedView>>);
static_assert(!HasMemberReserveHint<const std::ranges::enumerate_view<NonApproximatelySizedView>>);

struct ApproximatelySizedView : std::ranges::view_base {
  unsigned int size_;
  constexpr explicit ApproximatelySizedView(unsigned int hint) : size_(hint) {}
  constexpr auto begin() const { return forward_iterator<const int*>(globalBuff); }
  constexpr auto end() const { return forward_iterator<const int*>(globalBuff + 8); }
  constexpr unsigned int reserve_hint() const { return size_; }
};

static_assert(HasMemberReserveHint<std::ranges::enumerate_view<ApproximatelySizedView>>);
static_assert(HasMemberReserveHint<const std::ranges::enumerate_view<ApproximatelySizedView>>);

struct ApproximatelySizedNotConstView : std::ranges::view_base {
  unsigned int size_;
  constexpr explicit ApproximatelySizedNotConstView(unsigned int hint) : size_(hint) {}
  constexpr auto begin() const { return forward_iterator<const int*>(globalBuff); }
  constexpr auto end() const { return forward_iterator<const int*>(globalBuff + 8); }
  constexpr unsigned int reserve_hint() { return size_; }
};

static_assert(HasMemberReserveHint<std::ranges::enumerate_view<ApproximatelySizedNotConstView>>);
static_assert(!HasMemberReserveHint<const std::ranges::enumerate_view<ApproximatelySizedNotConstView>>);

constexpr bool test() {
  // Non-const and const are reserve_hint-able
  {
    auto view = std::views::enumerate(ApproximatelySizedView{5});
    assert(view.reserve_hint() == 5);
    assert(std::as_const(view).reserve_hint() == 5);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
