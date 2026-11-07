//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   V models only input_range:
//     friend constexpr difference_type operator-(default_sentinel_t y, const inner_iterator& x)
//       requires sized_sentinel_for<sentinel_t<V>, iterator_t<V>>;
//     friend constexpr difference_type operator-(const inner_iterator& x, default_sentinel_t y)
//       requires sized_sentinel_for<sentinel_t<V>, iterator_t<V>>;

#include <cassert>
#include <concepts>
#include <iterator>
#include <ranges>
#include <vector>

#include "test_iterators.h"

constexpr bool test() {
  std::vector<int> vector = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> single_vector  = {1};

  // Test `friend constexpr difference_type operator-(default_sentinel_t, const inner_iterator& x)`
  {
    // When range is general
    {
      std::ranges::chunk_view<
          std::ranges::subrange<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>>
          chunked =
              std::ranges::subrange(
                  cpp17_input_iterator<int*>(vector.data()),
                  sized_sentinel(cpp17_input_iterator<int*>(vector.data() + vector.size()))) |
              std::views::chunk(3);

      auto outer = chunked.begin();
      static_assert(std::same_as< decltype(std::default_sentinel - (*outer).begin()),
                                  typename std::ranges::iterator_t<
                                      std::ranges::range_reference_t<decltype(chunked)>>::difference_type>);
      assert(std::default_sentinel - (*outer).begin() == 3);
      ++outer;
      assert(std::default_sentinel - (*outer).begin() == 3);
      ++outer;
      assert(std::default_sentinel - (*outer).begin() == 2);
    }

    // When range is smaller than size
    {
      std::ranges::chunk_view<
          std::ranges::subrange<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>>
          chunked =
              std::ranges::subrange(
                  cpp17_input_iterator<int*>(single_vector.data()),
                  sized_sentinel(cpp17_input_iterator<int*>(single_vector.data() + single_vector.size()))) |
              std::views::chunk(3);

      auto outer = chunked.begin();
      static_assert(std::same_as< decltype(std::default_sentinel - (*outer).begin()),
                                  typename std::ranges::iterator_t<
                                      std::ranges::range_reference_t<decltype(chunked)>>::difference_type>);
      assert(std::default_sentinel - (*outer).begin() == 1);
    }
  }

  // Test `friend constexpr difference_type operator-(const inner_iterator& x, default_sentinel_t)`
  {
    // When range is general
    {
      std::ranges::chunk_view<
          std::ranges::subrange<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>>
          chunked =
              std::ranges::subrange(
                  cpp17_input_iterator<int*>(vector.data()),
                  sized_sentinel(cpp17_input_iterator<int*>(vector.data() + vector.size()))) |
              std::views::chunk(3);

      auto outer = chunked.begin();
      static_assert(std::same_as< decltype((*outer).begin() - std::default_sentinel),
                                  typename std::ranges::iterator_t<
                                      std::ranges::range_reference_t<decltype(chunked)>>::difference_type>);
      assert((*outer).begin() - std::default_sentinel == -3);
      ++outer;
      assert((*outer).begin() - std::default_sentinel == -3);
      ++outer;
      assert((*outer).begin() - std::default_sentinel == -2);
    }

    // When range is smaller than size
    {
      std::ranges::chunk_view<
          std::ranges::subrange<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>>
          chunked =
              std::ranges::subrange(
                  cpp17_input_iterator<int*>(single_vector.data()),
                  sized_sentinel(cpp17_input_iterator<int*>(single_vector.data() + single_vector.size()))) |
              std::views::chunk(3);

      auto outer = chunked.begin();
      static_assert(std::same_as< decltype((*outer).begin() - std::default_sentinel),
                                  typename std::ranges::iterator_t<
                                      std::ranges::range_reference_t<decltype(chunked)>>::difference_type>);
      assert((*outer).begin() - std::default_sentinel == -1);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
