//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto end() requires(!__simple_view<_View>)
// constexpr auto end() const requires(range<const _View>)

// Note: Checks here are augmented by checks in
// iterator/ctor.copy.pass.cpp.

#include <ranges>

#include "types.h"

template <class T>
concept HasConstEnd = requires(const T& ct) { ct.end(); };

template <class T>
concept HasEnd = requires(T& t) { t.end(); };

template <class T>
concept HasConstAndNonConstEnd =
    HasConstEnd<T> && requires(T& t, const T& ct) { requires !std::same_as<decltype(t.end()), decltype(ct.end())>; };

template <class T>
concept HasOnlyNonConstEnd = HasEnd<T> && !HasConstEnd<T>;

template <class T>
concept HasOnlyConstEnd = HasConstEnd<T> && !HasConstAndNonConstEnd<T>;

static_assert(HasOnlyNonConstEnd<std::ranges::stride_view<UnSimpleNoConstCommonView>>);
static_assert(HasOnlyConstEnd<std::ranges::stride_view<BasicTestView<int*, int*>>>);
static_assert(HasConstAndNonConstEnd<std::ranges::stride_view<UnsimpleConstView>>);

static_assert(simple_view<SimpleUnCommonConstView>);
static_assert(!std::ranges::common_range<SimpleUnCommonConstView>);

constexpr bool test_non_default_sentinel() {
  {
    static_assert(simple_view<BasicTestView<int*, int*>>);
    static_assert(std::ranges::common_range<BasicTestView<int*, int*>>);
    static_assert(std::ranges::sized_range<BasicTestView<int*, int*>>);
    static_assert(std::ranges::forward_range<BasicTestView<int*, int*>>);

    auto v  = BasicTestView<int*, int*>{nullptr, nullptr};
    auto sv = std::ranges::stride_view<BasicTestView<int*, int*>>(v, 1);
    static_assert(!std::is_same_v<std::default_sentinel_t, decltype(sv.end())>);
  }

  {
    static_assert(!simple_view<UnSimpleNoConstCommonView>);
    static_assert(std::ranges::common_range<UnSimpleNoConstCommonView>);
    static_assert(std::ranges::sized_range<UnSimpleNoConstCommonView>);
    static_assert(std::ranges::forward_range<UnSimpleNoConstCommonView>);

    auto v  = UnSimpleNoConstCommonView{};
    auto sv = std::ranges::stride_view<UnSimpleNoConstCommonView>(v, 1);
    static_assert(!std::is_same_v<std::default_sentinel_t, decltype(sv.end())>);
  }
  return true;
}

constexpr bool test_default_sentinel() {
  {
    static_assert(!simple_view<UnsimpleUnCommonConstView>);
    static_assert(!std::ranges::common_range<UnsimpleUnCommonConstView>);
    static_assert(std::ranges::sized_range<UnsimpleConstView>);
    static_assert(std::ranges::forward_range<UnsimpleConstView>);

    auto v  = UnsimpleUnCommonConstView{};
    auto sv = std::ranges::stride_view<UnsimpleUnCommonConstView>(v, 1);
    ASSERT_SAME_TYPE(std::default_sentinel_t, decltype(sv.end()));
  }

  {
    static_assert(simple_view<SimpleUnCommonConstView>);
    static_assert(!std::ranges::common_range<SimpleUnCommonConstView>);

    auto v  = SimpleUnCommonConstView{};
    auto sv = std::ranges::stride_view<SimpleUnCommonConstView>(v, 1);

    ASSERT_SAME_TYPE(std::default_sentinel_t, decltype(sv.end()));
  }
  return true;
}

int main(int, char**) {
  test_non_default_sentinel();
  test_default_sentinel();
  static_assert(test_non_default_sentinel());
  static_assert(test_default_sentinel());

  return 0;
}
