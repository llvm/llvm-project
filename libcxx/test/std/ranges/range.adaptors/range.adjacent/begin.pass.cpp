//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto begin() requires (!(simple-view<Views> && ...));
// constexpr auto begin() const requires (range<const Views> && ...);

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>
#include <utility>

#include "../range_adaptor_types.h"

template <class T>
concept HasConstBegin = requires(const T& ct) { ct.begin(); };

template <class T>
concept HasBegin = requires(T& t) { t.begin(); };

template <class T>
concept HasConstAndNonConstBegin = HasConstBegin<T> && requires(T& t, const T& ct) {
  requires !std::same_as<decltype(t.begin()), decltype(ct.begin())>;
};

template <class T>
concept HasOnlyNonConstBegin = HasBegin<T> && !HasConstBegin<T>;

template <class T>
concept HasOnlyConstBegin = HasConstBegin<T> && !HasConstAndNonConstBegin<T>;

struct NoConstBeginView : std::ranges::view_base {
  int* begin();
  int* end();
};

template <class Range, std::size_t N>
constexpr void test_one() {
  using View = std::ranges::adjacent_view<Range, N>;
  {
    int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    View v(Range{buffer});

    auto it    = v.begin();
    auto tuple = *it;
    assert(std::get<0>(tuple) == buffer[0]);
    if constexpr (N >= 2)
      assert(std::get<1>(tuple) == buffer[1]);
    if constexpr (N >= 3)
      assert(std::get<2>(tuple) == buffer[2]);

    auto cit    = std::as_const(v).begin();
    auto ctuple = *cit;
    assert(std::get<0>(ctuple) == buffer[0]);
    if constexpr (N >= 2)
      assert(std::get<1>(ctuple) == buffer[1]);
    if constexpr (N >= 3)
      assert(std::get<2>(ctuple) == buffer[2]);
    if constexpr (N >= 4)
      assert(std::get<3>(ctuple) == buffer[3]);
    if constexpr (N >= 5)
      assert(std::get<4>(ctuple) == buffer[4]);
  }

  {
    // empty range
    std::array<int, 0> buffer = {};
    View v(Range{buffer.data(), 0});
    auto it  = v.begin();
    auto cit = std::as_const(v).begin();
    assert(it == v.end());
    assert(cit == std::as_const(v).end());
  }

  if constexpr (N > 2) {
    // N greater than range size
    int buffer[2] = {1, 2};
    View v(Range{buffer});
    auto it  = v.begin();
    auto cit = std::as_const(v).begin();
    assert(it == v.end());
    assert(cit == std::as_const(v).end());
  }
}

template <std::size_t N>
constexpr void test() {
  {
    // Test with simple view
    test_one<SimpleCommon, N>();
    using View = std::ranges::adjacent_view<SimpleCommon, N>;
    static_assert(std::is_same_v<std::ranges::iterator_t<View>, std::ranges::iterator_t<const View>>);
  }

  {
    // Test with non-simple view
    test_one<NonSimpleCommon, N>();
    using View = std::ranges::adjacent_view<NonSimpleCommon, N>;
    static_assert(!std::is_same_v<std::ranges::iterator_t<View>, std::ranges::iterator_t<const View>>);
  }

  // Test with view that doesn't support const begin()
  using ViewWithNoConstBegin = std::ranges::adjacent_view<NoConstBeginView, N>;
  static_assert(!HasOnlyConstBegin<ViewWithNoConstBegin>);
  static_assert(HasOnlyNonConstBegin<ViewWithNoConstBegin>);
  static_assert(!HasConstAndNonConstBegin<ViewWithNoConstBegin>);
}

constexpr bool test() {
  test<1>();
  test<2>();
  test<3>();
  test<5>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
