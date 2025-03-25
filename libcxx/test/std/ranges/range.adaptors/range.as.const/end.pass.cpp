//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

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

static_assert(HasEnd<std::ranges::as_const_view<DefaultConstructibleView>>);
static_assert(HasEnd<const std::ranges::as_const_view<DefaultConstructibleView>>);
static_assert(HasEnd<std::ranges::as_const_view<NonConstView>>);
static_assert(!HasEnd<const std::ranges::as_const_view<NonConstView>>);

static_assert(std::is_same_v<decltype(std::declval<std::ranges::as_const_view<DefaultConstructibleView>>().end()),
                             std::const_iterator<int*>>);
static_assert(std::is_same_v<decltype(std::declval<const std::ranges::as_const_view<NonConstCommonRange>>().end()),
                             std::const_sentinel<sentinel_wrapper<int*>>>);

template <class Iter, class Sent>
constexpr void test_range() {
  int a[] = {1, 2};
  std::ranges::subrange range(Iter(std::begin(a)), Sent(Iter(std::end(a))));
  std::ranges::as_const_view view(std::move(range));
  std::same_as<std::const_sentinel<Sent>> decltype(auto) iter = view.end();
  assert(base(base(iter)) == std::end(a));
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
  int a[]    = {1, 2};
  auto range = WrapRange{Iter(a), Sent(Iter(a + 2))};
  const std::ranges::as_const_view view(std::move(range));
  std::same_as<std::const_sentinel<Sent>> decltype(auto) iter = view.end();
  assert(base(base(iter)) == std::end(a));
}

struct const_pointer_view : std::ranges::view_base {
  constexpr const int* begin() const { return {}; }
  constexpr const int* end() const { return {}; }
};
struct const_iterator_view : std::ranges::view_base {
  constexpr std::const_iterator<int*> begin() const { return {}; }
  constexpr std::const_iterator<int*> end() const { return {}; }
};

constexpr bool test() {
  test_range<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>();
  test_range<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>();
  test_range<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_range<cpp20_input_iterator<int*>, sized_sentinel<cpp20_input_iterator<int*>>>();

  types::for_each(types::forward_iterator_list<int*>{}, []<class Iter> {
    test_range<Iter, Iter>();
    test_range<Iter, sentinel_wrapper<Iter>>();
    test_range<Iter, sized_sentinel<Iter>>();
  });

  // check that with a const_iterator end() doesn't return std::const_iterator<const_iterator>
  {
    std::ranges::as_const_view view{const_pointer_view{}};
    std::same_as<const int*> decltype(auto) it = view.end();
    assert(it == nullptr);
  }
  {
    std::ranges::as_const_view view{const_iterator_view{}};
    std::same_as<std::const_iterator<int*>> decltype(auto) it = view.end();
    assert(it == std::const_iterator<int*>{});
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
