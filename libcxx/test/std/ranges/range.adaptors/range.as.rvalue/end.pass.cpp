//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto end()
// constexpr auto end() const

#include <array>
#include <cassert>
#include <ranges>

#include "test_iterators.h"

struct DefaultConstructibleView : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

struct CVCallView : std::ranges::view_base {
  mutable bool const_called = false;
  mutable int i[1];
  constexpr int* begin() {
    const_called = false;
    return i;
  }

  constexpr int* begin() const {
    const_called = true;
    return i;
  }

  constexpr int* end() {
    const_called = false;
    return i + 1;
  }

  constexpr int* end() const {
    const_called = true;
    return i + 1;
  }
};

struct NonConstCommonRange : std::ranges::view_base {
  int* begin();
  int* end();

  int* begin() const;
  sentinel_wrapper<int*> end() const;
};

struct NonConstView : std::ranges::view_base {
  int* begin();
  int* end();
};

template <class T>
concept HasEnd = requires(T t) { t.end(); };

static_assert(HasEnd<std::ranges::as_rvalue_view<DefaultConstructibleView>>);
static_assert(HasEnd<const std::ranges::as_rvalue_view<DefaultConstructibleView>>);
static_assert(HasEnd<std::ranges::as_rvalue_view<NonConstView>>);
static_assert(!HasEnd<const std::ranges::as_rvalue_view<NonConstView>>);

static_assert(std::is_same_v<decltype(std::declval<std::ranges::as_rvalue_view<DefaultConstructibleView>>().end()),
                             std::move_iterator<int*>>);
static_assert(std::is_same_v<decltype(std::declval<const std::ranges::as_rvalue_view<NonConstCommonRange>>().end()),
                             std::move_sentinel<sentinel_wrapper<int*>>>);

template <class Iter, class Sent, bool is_common>
constexpr void test_range() {
  using Expected = std::conditional_t<is_common, std::move_iterator<Sent>, std::move_sentinel<Sent>>;
  int a[]        = {1, 2};
  std::ranges::subrange range(Iter(std::begin(a)), Sent(Iter(std::end(a))));
  std::ranges::as_rvalue_view view(std::move(range));
  std::same_as<Expected> decltype(auto) iter = view.end();
  assert(base(base(iter.base())) == std::end(a));
}

template <class Iter, class Sent>
class WrapRange {
  Iter iter_;
  Sent sent_;

public:
  constexpr WrapRange(Iter iter, Sent sent) : iter_(std::move(iter)), sent_(std::move(sent)) {}

  constexpr Iter begin() const { return iter_; }
  constexpr Sent end() const { return sent_; }
};

template <class Iter, class Sent>
WrapRange(Iter, Sent) -> WrapRange<Iter, Sent>;

template <class Iter, class Sent, bool is_common>
constexpr void test_const_range() {
  using Expected = std::conditional_t<is_common, std::move_iterator<Sent>, std::move_sentinel<Sent>>;
  int a[]        = {1, 2};
  auto range = WrapRange{Iter(a), Sent(Iter(a + 2))};
  const std::ranges::as_rvalue_view view(std::move(range));
  std::same_as<Expected> decltype(auto) iter = view.end();
  assert(base(base(iter.base())) == std::end(a));
}

struct move_iterator_view : std::ranges::view_base {
  constexpr std::move_iterator<int*> begin() const { return {}; }
  constexpr std::move_iterator<int*> end() const { return {}; }
};

constexpr bool test() {
  test_range<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>, false>();
  test_range<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>, false>();
  test_range<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>, false>();
  test_range<cpp20_input_iterator<int*>, sized_sentinel<cpp20_input_iterator<int*>>, false>();

  meta::for_each(meta::forward_iterator_list<int*>{}, []<class Iter> {
    test_range<Iter, Iter, true>();
    test_range<Iter, sentinel_wrapper<Iter>, false>();
    test_range<Iter, sized_sentinel<Iter>, false>();
  });

  {
    std::ranges::as_rvalue_view view(CVCallView{});
    (void)view.end();
    assert(view.base().const_called);
  }

  { // check that with a std::move_iterator begin() doesn't return move_iterator<move_iterator<T>>
    std::ranges::as_rvalue_view view{move_iterator_view{}};
    std::same_as<std::move_iterator<int*>> decltype(auto) it = view.end();
    assert(it == std::move_iterator<int*>{});
  }

  return true;
}

int main(int, char**) {
  test();

// gcc cannot have mutable member in constant expression
#if !defined(TEST_COMPILER_GCC)
  static_assert(test());
#endif

  return 0;
}
