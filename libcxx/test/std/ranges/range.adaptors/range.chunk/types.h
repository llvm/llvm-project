//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_CHUNK_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_CHUNK_TYPES_H

#include <cstddef>
#include <iterator>
#include <ranges>
#include <span>
#include <vector>

// input_span

template <class T>
struct input_span : std::span<T> {
  struct iterator : std::span<T>::iterator {
    using iterator_concept = std::input_iterator_tag;
    constexpr iterator()   = default;
    constexpr iterator(std::span<T>::iterator i) : std::span<T>::iterator(i) {}
    constexpr auto operator*() const { return std::span<T>::iterator::operator*(); }
    friend constexpr auto operator+(iterator, std::span<T>::difference_type) = delete;
    friend constexpr auto operator+(std::span<T>::difference_type, iterator) = delete;
    friend constexpr auto operator-(iterator, std::span<T>::difference_type) = delete;
    friend constexpr auto operator-(std::span<T>::difference_type, iterator) = delete;
    friend constexpr iterator& operator++(iterator& self) {
      ++static_cast<std::span<T>::iterator&>(self);
      return self;
    }
    friend constexpr void operator++(iterator& self, int) { ++self; }
    friend constexpr iterator& operator--(iterator&) = delete;
    friend constexpr void operator--(iterator&, int) = delete;
  };

  using std::span<T>::span;
  constexpr iterator begin() { return iterator(std::span<T>::begin()); }
  constexpr iterator end() { return iterator(std::span<T>::end()); }
};

template <class T>
input_span(T*, std::ptrdiff_t) -> input_span<T>;

template <class T>
inline constexpr bool std::ranges::enable_view<input_span<T>> = true;

static_assert(std::ranges::input_range<input_span<int>> && !std::ranges::forward_range<input_span<int>> &&
              std::ranges::view<input_span<int>>);

#endif
