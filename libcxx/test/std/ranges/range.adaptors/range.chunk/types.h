//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_CHUNK_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_CHUNK_TYPES_H

#include <forward_list>
#include <ranges>
#include <vector>

template <std::ranges::view View>
struct exactly_input_view : View, std::ranges::view_interface<exactly_input_view<View>> {
  struct iterator : std::ranges::iterator_t<View> {
    using iterator_concept = std::input_iterator_tag;
    constexpr iterator()   = default;
    constexpr iterator(std::ranges::iterator_t<View> i) : std::ranges::iterator_t<View>(i) {}
    constexpr auto operator*() const { return std::ranges::iterator_t<View>::operator*(); }
    friend constexpr void operator+(auto&&...) = delete;
    friend constexpr void operator-(auto&&...) = delete;
    friend constexpr iterator& operator++(iterator& self) {
      self.std::ranges::template iterator_t<View>::operator++();
      return self;
    }
    friend constexpr void operator++(iterator& self, int) { ++self; }
    friend constexpr void operator--(auto&&...) = delete;
  };

  constexpr iterator begin(this auto&& self) { return iterator(self.View::begin()); }
  constexpr iterator end(this auto&& self) { return iterator(self.View::end()); }
};

template <std::ranges::view View>
struct not_sized_view : View, std::ranges::view_interface<not_sized_view<View>> {
  struct iterator : std::ranges::iterator_t<View> {
    using iterator_concept = std::bidirectional_iterator_tag;
    constexpr iterator()   = default;
    constexpr iterator(std::ranges::iterator_t<View> i) : std::ranges::iterator_t<View>(i) {}
    friend constexpr void operator-(iterator, iterator) = delete;
    friend constexpr iterator& operator++(iterator& self) {
      self.std::ranges::template iterator_t<View>::operator++();
      return self;
    }
    friend constexpr iterator operator++(iterator& self, int) { return ++self; }
    friend constexpr iterator& operator--(iterator& self) {
      self.std::ranges::template iterator_t<View>::operator--();
      return self;
    }
    friend constexpr iterator operator--(iterator& self, int) { return --self; }
  };

  constexpr iterator begin(this auto&& self) { return iterator(self.View::begin()); }
  constexpr iterator end(this auto&& self) { return iterator(self.View::end()); }
  constexpr auto size() const = delete;
};

template <std::ranges::view View>
inline constexpr bool std::ranges::disable_sized_range<not_sized_view<View>> = true;

#endif