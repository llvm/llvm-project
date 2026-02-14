//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// constexpr auto end() requires(!__simple_view<_View>)
// constexpr auto end() const requires(range<const _View>)

// Note: Checks here are augmented by checks in
// iterator/ctor.copy.pass.cpp.

#include <ranges>

#include "test_iterators.h"
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
static_assert(HasOnlyConstEnd<std::ranges::stride_view<SimpleCommonConstView>>);
static_assert(HasConstAndNonConstEnd<std::ranges::stride_view<UnSimpleConstView>>);

constexpr bool test_non_default_sentinel() {
  {
    const int data[3] = {1, 2, 3};
    // A const, simple, common-, sized- and forward-range
    // Note: sized because it is possible to get a difference between its
    // beginning and its end.
    auto v  = BasicTestView<const int*, const int*>{data, data + 3};
    auto sv = std::ranges::stride_view<BasicTestView<const int*, const int*>>(v, 1);
    static_assert(!std::is_same_v<std::default_sentinel_t, decltype(sv.end())>);
  }

  {
    int data[3] = {1, 2, 3};
    // ForwardTestView is not sized and not bidirectional, but it is common.
    // Note: It is not sized because BasicTestView has no member function named size (by default)
    // and nor is it possible to get a difference between its beginning and its end.
    using ForwardTestView = BasicTestView<forward_iterator<int*>, forward_iterator<int*>>;

    auto v  = ForwardTestView{forward_iterator(data), forward_iterator(data + 3)};
    auto sv = std::ranges::stride_view<ForwardTestView>(v, 1);
    static_assert(!std::is_same_v<std::default_sentinel_t, decltype(sv.end())>);
  }

  {
    // TODO: Start here.
    static_assert(!simple_view<UnSimpleNoConstCommonView>);
    static_assert(std::ranges::common_range<UnSimpleNoConstCommonView>);
    static_assert(std::ranges::sized_range<UnSimpleNoConstCommonView>);
    static_assert(std::ranges::forward_range<UnSimpleNoConstCommonView>);

    // An unconst, unsimple, common-, sized- and forward-range
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
    static_assert(std::ranges::sized_range<UnSimpleConstView>);
    static_assert(std::ranges::forward_range<UnSimpleConstView>);

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
