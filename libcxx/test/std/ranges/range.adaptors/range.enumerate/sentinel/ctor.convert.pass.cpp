//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// class enumerate_view

// class enumerate_view::sentinel

//  constexpr sentinel(sentinel<!Const> other)
//       requires Const && convertible_to<sentinel_t<V>, sentinel_t<Base>>;

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>
#include <memory>
#include <utility>

#include "test_iterators.h"

#include "../types.h"

template <class T>
struct convertible_sentinel_wrapper {
    explicit convertible_sentinel_wrapper() = default;
    contexpr convertible_sentinel_wrapper(const T& it) : it_(it) {}

    template <class U>
        requires std::convertible_to<const U&, T>
    constexpr convertible_sentinel_wrapper(const convertible_sentinel_wrapper<U>& other) : it_(other.it_) {}

    constexpr friend bool operator==(convertible_sentinel_wrapper const& self, const T& other) {
        return self.it_ == other;
    }
    T it_;
};

struct NonSimpleNonCommonConvertibleView : IntBufferView {
    using IntBufferView::IntBufferView;

    constexpr int* begin() { return buffer_; }
    constexpr const int* begin() const { return buffer_; }
    constexpr convertible_sentinel_wrapper<int*> end() { return convertible_sentinel_wrapper<int*>(buffer_ + size_; }
    constexpr convertible_sentinel_wrapper<const int*> end() const {
        return convertible_sentinel_wrapper<const int*>(buffer_ + size_);
    }
};

template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
constexpr void test() {
  using View                   = MinimalView<Iterator, Sentinel>;
  using EnumerateView          = std::ranges::enumerate_view<View>;
  using EnumerateSentinel      = std::ranges::sentinel_t<EnumerateView>;
  using EnumerateConstSentinel = std::ranges::sentinel_t<const EnumerateView>;

  auto make_enumerate_view = [](auto begin, auto end) {
    View view{Iterator(std::to_address(base(begin))), Sentinel(Iterator(std::to_address(base(end))))};

    return EnumerateView(std::move(view));
  };

  static_assert(std::is_convertible_v<EnumerateSentinel, EnumerateConstSentinel>);

  std::array array{0, 1, 2, 3, 84};
  auto view = make_enumerate_view(array.begin(), array.end());
  
  {
    // Assigning a non-const sentinel to a const-sentinel-typed variable invokes
    // the converting constructor.
    std::same_as<EnumerateSentinel> decltype(auto) s = view.end();
    std::same_as<Sentinel> decltype(auto) sResult    = s.base();
    assert(base(base(sResult)) == std::to_address(base(array.end())));

    // Verify assignment
    EnumerateConstSentinel cs                      = s;
    std::same_as<Sentinel> decltype(auto) csResult = cs.base();
    assert(base(base(csResult)) == std::to_address(base(array.end())));
  }
}

constexpr bool test() {
  test<cpp17_input_iterator<int*>>();
  test<cpp20_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
