//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// ranges

// std::views::stride_view

#include "__ranges/stride_view.h"
#include "test.h"

// Do not use for execution -- only useful for testing compilation/type conditions.
template <typename T>
struct non_input_iterator {
  struct __invalid_iterator_tag {};
  using value_type       = T;
  using difference_type  = int;
  using iterator_concept = __invalid_iterator_tag;

  constexpr non_input_iterator& operator++() { return *this; }
  constexpr void operator++(int) {}
  constexpr T operator*() const { return T{}; }

  friend constexpr bool operator==(const non_input_iterator&, const non_input_iterator&) { return true; }
};

template <typename T>
inline constexpr bool std::ranges::enable_borrowed_range<non_input_iterator<T>> = true;

class almost_input_range : public std::ranges::view_base {
public:
  constexpr auto begin() const { return non_input_iterator<int>{}; }
  constexpr auto end() const { return non_input_iterator<int>{}; }
};

constexpr bool test() {
  // Ensure that the almost_input_range is a valid range.
  static_assert(std::ranges::range<almost_input_range>);
  // Ensure that the non_input_iterator is, well, not an input iterator.
  static_assert(!std::input_iterator<non_input_iterator<int>>);

  static_assert(!CanStrideView<almost_input_range, 1>, "A non input range cannot be the subject of a stride view.");
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
