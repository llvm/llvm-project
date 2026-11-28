//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

//   template<input_range V>
//     requires view<V>
//   template<bool Const>
//   class as_input_view<V>::iterator

//    constexpr iterator(iterator<!Const> i)
//      requires Const && convertible_to<iterator_t<V>, iterator_t<Base>>;

#include <cassert>
#include <concepts>
#include <cstdint>
#include <ranges>
#include <type_traits>
#include <utility>

#include "test_iterators.h"
#include "test_range.h"

class NonSimpleView : public std::ranges::view_interface<NonSimpleView> {
  int* data_;
  std::size_t size_;

public:
  constexpr NonSimpleView(int* data, std::size_t size) : data_(data), size_(size) {}

  constexpr int* begin() { return data_; }
  constexpr const int* begin() const { return data_; }
  constexpr int* end() { return data_ + size_; }
  constexpr const int* end() const { return data_ + size_; }
  constexpr std::size_t size() const { return size_; }
};
static_assert(!simple_view<NonSimpleView>);

constexpr bool test() {
  int arr[] = {94};
  NonSimpleView range{arr, 1};

  std::ranges::as_input_view<NonSimpleView> view{range};

  using IteratorT      = std::ranges::iterator_t<decltype(view)>;
  using ConstIteratorT = std::ranges::iterator_t<const decltype(view)>;

  static_assert(!std::same_as<IteratorT, ConstIteratorT>);
  static_assert(std::convertible_to<IteratorT, ConstIteratorT>);
  static_assert(std::constructible_from<ConstIteratorT, IteratorT&&>);

  IteratorT it = view.begin();
  ConstIteratorT const_it{std::move(it)};

  std::same_as<const int*> decltype(auto) base_it = std::move(const_it).base();
  assert(base(base_it) == arr);
  assert(*base_it == 94);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
