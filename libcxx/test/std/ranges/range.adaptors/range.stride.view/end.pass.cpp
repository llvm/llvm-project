//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto end() requires(!simple-view<V>)
// constexpr auto end() const requires(range<const V>)

// Note: Checks here are augmented by checks in
// iterator/ctor.copy.pass.cpp.

#include <ranges>
#include <cassert>
#include <type_traits>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

// A view that is a common forward range when const, but NOT a common range when non-const.
struct CommonForwardOnlyWhenConst : std::ranges::view_base {
  int* data_;
  int size_;

  constexpr CommonForwardOnlyWhenConst(int* d, int s) : data_(d), size_(s) {}

  CommonForwardOnlyWhenConst(CommonForwardOnlyWhenConst&&)            = default;
  CommonForwardOnlyWhenConst& operator=(CommonForwardOnlyWhenConst&&) = default;

  // Non-const: not a common range (iterator and sentinel are different types).
  constexpr forward_iterator<int*> begin() { return forward_iterator<int*>(data_); }
  constexpr sentinel_wrapper<forward_iterator<int*>> end() {
    return sentinel_wrapper<forward_iterator<int*>>(forward_iterator<int*>(data_ + size_));
  }

  // Const: a common forward range (begin and end return the same type).
  constexpr forward_iterator<const int*> begin() const { return forward_iterator<const int*>(data_); }
  constexpr forward_iterator<const int*> end() const { return forward_iterator<const int*>(data_ + size_); }
};

static_assert(std::ranges::range<CommonForwardOnlyWhenConst>);
static_assert(std::ranges::range<const CommonForwardOnlyWhenConst>);
static_assert(std::ranges::forward_range<const CommonForwardOnlyWhenConst>);
static_assert(!std::ranges::bidirectional_range<const CommonForwardOnlyWhenConst>);
static_assert(std::ranges::common_range<const CommonForwardOnlyWhenConst>);
static_assert(!std::ranges::common_range<CommonForwardOnlyWhenConst>);
static_assert(std::ranges::view<CommonForwardOnlyWhenConst>);

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

constexpr bool test() {
  {
    // A const, simple, common-, sized- and forward-range.
    // Note: sized because it is possible to get a difference between its
    // beginning and its end.
    const int data[] = {1, 2, 3};
    auto v           = BasicTestView<const int*, const int*>{data, data + 3};
    auto sv          = std::ranges::stride_view(v, 1);
    static_assert(!std::is_same_v<std::default_sentinel_t, decltype(sv.end())>);

    // Verify actual end behavior: iterating reaches end.
    auto it = sv.begin();
    ++it;
    ++it;
    ++it;
    assert(it == sv.end());
  }
  {
    // ForwardTestView is not sized and not bidirectional, but it is common.
    // Note: It is not sized because BasicTestView has no member function named size (by default)
    // and nor is it possible to get a difference between its beginning and its end.
    int data[]            = {1, 2, 3};
    using ForwardTestView = BasicTestView<forward_iterator<int*>, forward_iterator<int*>>;
    auto v                = ForwardTestView{forward_iterator(data), forward_iterator(data + 3)};
    auto sv               = std::ranges::stride_view(v, 1);
    static_assert(!std::is_same_v<std::default_sentinel_t, decltype(sv.end())>);

    auto it = sv.begin();
    ++it;
    ++it;
    ++it;
    assert(it == sv.end());
  }
  {
    // A non-const, non-simple, common-, sized- and forward-range.
    static_assert(!simple_view<UnSimpleNoConstCommonView>);
    static_assert(std::ranges::common_range<UnSimpleNoConstCommonView>);
    static_assert(std::ranges::sized_range<UnSimpleNoConstCommonView>);
    static_assert(std::ranges::forward_range<UnSimpleNoConstCommonView>);

    auto sv = std::ranges::stride_view<UnSimpleNoConstCommonView>(UnSimpleNoConstCommonView{}, 1);
    static_assert(!std::is_same_v<std::default_sentinel_t, decltype(sv.end())>);
  }
  {
    // Uncommon range -> returns default_sentinel.
    static_assert(!simple_view<UnsimpleUnCommonConstView>);
    static_assert(!std::ranges::common_range<UnsimpleUnCommonConstView>);

    auto sv = std::ranges::stride_view<UnsimpleUnCommonConstView>(UnsimpleUnCommonConstView{}, 1);
    ASSERT_SAME_TYPE(std::default_sentinel_t, decltype(sv.end()));
  }
  {
    // Simple, uncommon range -> returns default_sentinel.
    static_assert(simple_view<SimpleUnCommonConstView>);
    static_assert(!std::ranges::common_range<SimpleUnCommonConstView>);

    auto sv = std::ranges::stride_view<SimpleUnCommonConstView>(SimpleUnCommonConstView{}, 1);
    ASSERT_SAME_TYPE(std::default_sentinel_t, decltype(sv.end()));
  }
  {
    // Verify stride > 1 with end(): iterating produces correct elements and terminates.
    int data[] = {10, 20, 30, 40, 50};
    auto v     = BasicTestView<int*, int*>{data, data + 5};
    auto sv    = std::ranges::stride_view(v, 2);

    auto it = sv.begin();
    assert(*it == 10);
    ++it;
    assert(*it == 30);
    ++it;
    assert(*it == 50);
    ++it;
    assert(it == sv.end());
  }
  {
    // Verify end() with stride that doesn't evenly divide the range.
    int data[] = {1, 2, 3, 4, 5, 6, 7};
    auto v     = BasicTestView<int*, int*>{data, data + 7};
    auto sv    = std::ranges::stride_view(v, 3);

    auto it = sv.begin();
    assert(*it == 1);
    ++it;
    assert(*it == 4);
    ++it;
    assert(*it == 7);
    ++it;
    assert(it == sv.end());
  }
  {
    // Regression test: end() const should use common_range<const _View>, not common_range<_View>.
    // CommonForwardOnlyWhenConst is common + forward-only when const, but NOT common when non-const.
    // With the bug, end() const would incorrectly check common_range<_View> (false), skip the second
    // branch, and return default_sentinel. With the fix, it checks common_range<const _View> (true)
    // and returns an iterator.
    int data[]      = {1, 2, 3, 4, 5};
    auto v          = CommonForwardOnlyWhenConst(data, 5);
    auto sv         = std::ranges::stride_view(std::move(v), 2);
    const auto& csv = sv;

    // The key assertion: end() on the const stride_view must NOT return default_sentinel_t.
    static_assert(!std::is_same_v<std::default_sentinel_t, decltype(csv.end())>);

    // Verify iteration actually works and reaches end.
    auto it = csv.begin();
    assert(*it == 1);
    ++it;
    assert(*it == 3);
    ++it;
    assert(*it == 5);
    ++it;
    assert(it == csv.end());
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
