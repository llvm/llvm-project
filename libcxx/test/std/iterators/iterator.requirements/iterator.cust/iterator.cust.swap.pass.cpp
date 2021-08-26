//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class I>
// unspecified iter_swap;

#include <iterator>

#include <array>
#include <cassert>

#include "./unqualified_lookup_wrapper.h"
#include "test_iterators.h"

using IterSwapT = decltype(std::ranges::iter_swap);

static_assert(std::semiregular<std::remove_cv_t<IterSwapT>>);

struct HasIterSwap {
  int &value_;
  explicit HasIterSwap(int &value) : value_(value) { assert(value == 0); }

  friend constexpr void iter_swap(HasIterSwap& a, HasIterSwap& b) {
    a.value_ = 1;
    b.value_ = 1;
  }
  friend constexpr void iter_swap(HasIterSwap& a, int& b) {
    a.value_ = 2;
    b = 2;
  }
};

static_assert( std::is_invocable_v<IterSwapT, HasIterSwap&, HasIterSwap&>);
static_assert( std::is_invocable_v<IterSwapT, HasIterSwap&, int&>);
static_assert(!std::is_invocable_v<IterSwapT, int&, HasIterSwap&>);

static_assert( std::is_invocable_v<IterSwapT&, HasIterSwap&, HasIterSwap&>);
static_assert( std::is_invocable_v<IterSwapT&, HasIterSwap&, int&>);
static_assert(!std::is_invocable_v<IterSwapT&, int&, HasIterSwap&>);

static_assert( std::is_invocable_v<IterSwapT&&, HasIterSwap&, HasIterSwap&>);
static_assert( std::is_invocable_v<IterSwapT&&, HasIterSwap&, int&>);
static_assert(!std::is_invocable_v<IterSwapT&&, int&, HasIterSwap&>);

struct NodiscardIterSwap {
  [[nodiscard]] friend int iter_swap(NodiscardIterSwap&, NodiscardIterSwap&) { return 0; }
};

void ensureVoidCast(NodiscardIterSwap& a, NodiscardIterSwap& b) { std::ranges::iter_swap(a, b); }

struct HasRangesSwap {
  int &value_;
  explicit HasRangesSwap(int &value) : value_(value) { assert(value == 0); }

  friend constexpr void swap(HasRangesSwap& a, HasRangesSwap& b) {
    a.value_ = 1;
    b.value_ = 1;
  }
  friend constexpr void swap(HasRangesSwap& a, int& b) {
    a.value_ = 2;
    b = 2;
  }
};

struct HasRangesSwapWrapper {
  using value_type = HasRangesSwap;

  HasRangesSwap &value_;
  explicit HasRangesSwapWrapper(HasRangesSwap &value) : value_(value) {}

  HasRangesSwap& operator*() const { return value_; }
};

static_assert( std::is_invocable_v<IterSwapT, HasRangesSwapWrapper&, HasRangesSwapWrapper&>);
// Does not satisfy swappable_with, even though swap(X, Y) is valid.
static_assert(!std::is_invocable_v<IterSwapT, HasRangesSwapWrapper&, int&>);
static_assert(!std::is_invocable_v<IterSwapT, int&, HasRangesSwapWrapper&>);

struct B;

struct A {
  bool value = false;
  A& operator=(const B&) {
    value = true;
    return *this;
  };
};

struct B {
  bool value = false;
  B& operator=(const A&) {
    value = true;
    return *this;
  };
};

struct MoveOnly2;

struct MoveOnly1 {
  bool value = false;

  MoveOnly1() = default;
  MoveOnly1(MoveOnly1&&) = default;
  MoveOnly1& operator=(MoveOnly1&&) = default;
  MoveOnly1(const MoveOnly1&) = delete;
  MoveOnly1& operator=(const MoveOnly1&) = delete;

  MoveOnly1& operator=(MoveOnly2 &&) {
    value = true;
    return *this;
  };
};

struct MoveOnly2 {
  bool value = false;

  MoveOnly2() = default;
  MoveOnly2(MoveOnly2&&) = default;
  MoveOnly2& operator=(MoveOnly2&&) = default;
  MoveOnly2(const MoveOnly2&) = delete;
  MoveOnly2& operator=(const MoveOnly2&) = delete;

  MoveOnly2& operator=(MoveOnly1 &&) {
    value = true;
    return *this;
  };
};

int main(int, char**) {
  {
    int value1 = 0;
    int value2 = 0;
    HasIterSwap a(value1), b(value2);
    std::ranges::iter_swap(a, b);
    assert(value1 == 1 && value2 == 1);
  }

  {
    int value1 = 0;
    int value2 = 0;
    HasRangesSwap c(value1), d(value2);
    HasRangesSwapWrapper cWrapper(c), dWrapper(d);
    std::ranges::iter_swap(cWrapper, dWrapper);
    assert(value1 == 1 && value2 == 1);
  }

  {
    int value1 = 0;
    int value2 = 0;
    HasRangesSwap c(value1), d(value2);
    std::ranges::iter_swap(HasRangesSwapWrapper(c), HasRangesSwapWrapper(d));
    assert(value1 == 1 && value2 == 1);
  }

  {
    A e; B f;
    A *ePtr = &e;
    B *fPtr = &f;
    std::ranges::iter_swap(ePtr, fPtr);
    assert(e.value && f.value);
  }

  {
    MoveOnly1 g; MoveOnly2 h;
    std::ranges::iter_swap(&g, &h);
    assert(g.value && h.value);
  }

  {
    auto arr = std::array<move_tracker, 2>();
    std::ranges::iter_swap(arr.begin(), arr.begin() + 1);
    assert(arr[0].moves() == 1 && arr[1].moves() == 2);
  }

  {
    int buff[2] = {1, 2};
    std::ranges::iter_swap(buff + 0, buff + 1);
    assert(buff[0] == 2 && buff[1] == 1);

    std::ranges::iter_swap(cpp20_input_iterator(buff), cpp20_input_iterator(buff + 1));
    assert(buff[0] == 1 && buff[1] == 2);

    std::ranges::iter_swap(cpp17_input_iterator(buff), cpp17_input_iterator(buff + 1));
    assert(buff[0] == 2 && buff[1] == 1);

    std::ranges::iter_swap(forward_iterator(buff), forward_iterator(buff + 1));
    assert(buff[0] == 1 && buff[1] == 2);

    std::ranges::iter_swap(bidirectional_iterator(buff), bidirectional_iterator(buff + 1));
    assert(buff[0] == 2 && buff[1] == 1);

    std::ranges::iter_swap(random_access_iterator(buff), random_access_iterator(buff + 1));
    assert(buff[0] == 1 && buff[1] == 2);

    std::ranges::iter_swap(contiguous_iterator(buff), contiguous_iterator(buff + 1));
    assert(buff[0] == 2 && buff[1] == 1);
  }

  return 0;
}
