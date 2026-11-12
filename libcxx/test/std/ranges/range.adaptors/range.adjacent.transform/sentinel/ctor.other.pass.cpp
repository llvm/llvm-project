//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr sentinel(sentinel<!Const> s);
//   requires Const && convertible_to<inner-sentinel<false>, inner-sentinel<Const>>;

#include <cassert>
#include <ranges>
#include <utility>

#include "../helpers.h"
#include "../../range_adaptor_types.h"

template <class T>
struct convertible_sentinel_wrapper {
  explicit convertible_sentinel_wrapper() = default;
  constexpr convertible_sentinel_wrapper(const T& it) : it_(it) {}

  template <class U>
    requires std::convertible_to<const U&, T>
  constexpr convertible_sentinel_wrapper(const convertible_sentinel_wrapper<U>& other) : it_(other.it_) {}

  constexpr friend bool operator==(convertible_sentinel_wrapper const& self, const T& other) {
    return self.it_ == other;
  }
  T it_;
};

struct SentinelConvertibleView : IntBufferView {
  using IntBufferView::IntBufferView;

  constexpr int* begin() { return buffer_; }
  constexpr const int* begin() const { return buffer_; }
  constexpr convertible_sentinel_wrapper<int*> end() { return convertible_sentinel_wrapper<int*>(buffer_ + size_); }
  constexpr convertible_sentinel_wrapper<const int*> end() const {
    return convertible_sentinel_wrapper<const int*>(buffer_ + size_);
  }
};

static_assert(!std::ranges::common_range<SentinelConvertibleView>);
static_assert(std::convertible_to<std::ranges::sentinel_t<SentinelConvertibleView>,
                                  std::ranges::sentinel_t<SentinelConvertibleView const>>);
static_assert(!simple_view<SentinelConvertibleView>);

struct SentinelNonConvertibleView : IntBufferView {
  using IntBufferView::IntBufferView;

  constexpr int* begin() { return buffer_; }
  constexpr const int* begin() const { return buffer_; }
  constexpr sentinel_wrapper<int*> end() { return sentinel_wrapper<int*>(buffer_ + size_); }
  constexpr sentinel_wrapper<const int*> end() const { return sentinel_wrapper<const int*>(buffer_ + size_); }
};

static_assert(!std::ranges::common_range<SentinelNonConvertibleView>);
static_assert(!std::convertible_to<std::ranges::sentinel_t<SentinelNonConvertibleView>,
                                   std::ranges::sentinel_t<SentinelNonConvertibleView const>>);
static_assert(!simple_view<SentinelNonConvertibleView>);

template <std::size_t N, class Fn>
constexpr void test() {
  using View = std::ranges::adjacent_transform_view<SentinelConvertibleView, Fn, N>;
  static_assert(!std::ranges::common_range<View>);

  using Sent      = std::ranges::sentinel_t<View>;
  using ConstSent = std::ranges::sentinel_t<const View>;
  static_assert(!std::is_same_v<Sent, ConstSent>);

  {
    // implicitly convertible
    static_assert(std::convertible_to<Sent, ConstSent>);
  }
  {
    // !Const
    static_assert(!std::convertible_to<ConstSent, Sent>);
  }
  {
    // !convertible_to<iterator_t<V>, iterator_t<Base>>
    using V2 = std::ranges::adjacent_transform_view<SentinelNonConvertibleView, Fn, N>;
    static_assert(!std::convertible_to<std::ranges::sentinel_t<V2>, std::ranges::sentinel_t<const V2>>);
  }

  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    View v{SentinelConvertibleView{buffer}, Fn{}};
    Sent sent1      = v.end();
    ConstSent sent2 = sent1;

    assert(v.begin() != sent2);
    assert(std::as_const(v).begin() != sent2);
    assert(v.begin() + (10 - N) == sent2);
    assert(std::as_const(v).begin() + (10 - N) == sent2);
  }
}

template <std::size_t N>
constexpr void test() {
  test<N, MakeTuple>();
  test<N, Tie>();
  test<N, GetFirst>();
  test<N, Multiply>();
}

constexpr bool test() {
  test<1>();
  test<2>();
  test<3>();
  test<4>();
  test<5>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
