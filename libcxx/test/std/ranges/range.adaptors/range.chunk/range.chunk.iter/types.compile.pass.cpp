//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   V models forward_range:
//     using iterator_category = ...;
//     using iterator_concept = ...;
//     using value_type = decltype(views::take(subrange(current_, end_), n_));
//     using difference_type = range_difference_t<Base>;

#include <concepts>
#include <iterator>
#include <ranges>

#include "test_iterators.h"

constexpr void test() {
  // Test `using iterator_category = ...`
  // Test `using iterator_concept = ...`
  // Test `using value_type = decltype(view::take(subrange(current_, end_), n_))`
  // Test `using difference_type = range_difference_t<Base>`
  {
    // forward_iterator
    {
      static_assert(
          std::same_as< typename std::ranges::iterator_t< std::ranges::chunk_view<
                            std::ranges::subrange<forward_iterator<int*>, sentinel_wrapper<forward_iterator<int*>>>>>::
                            iterator_category,
                        std::input_iterator_tag>);
      static_assert(
          std::same_as< typename std::ranges::iterator_t< std::ranges::chunk_view<
                            std::ranges::subrange<forward_iterator<int*>, sentinel_wrapper<forward_iterator<int*>>>>>::
                            iterator_concept,
                        std::forward_iterator_tag>);
      static_assert(
          std::same_as<
              typename std::ranges::iterator_t< std::ranges::chunk_view<
                  std::ranges::subrange<forward_iterator<int*>, sentinel_wrapper<forward_iterator<int*>>>>>::value_type,
              std::ranges::range_value_t<std::ranges::chunk_view<
                  std::ranges::subrange<forward_iterator<int*>, sentinel_wrapper<forward_iterator<int*>>>>>>);
      static_assert(
          std::same_as<typename std::ranges::iterator_t< std::ranges::chunk_view<
                           std::ranges::subrange<forward_iterator<int*>, sentinel_wrapper<forward_iterator<int*>>>>>::
                           difference_type,
                       std::ranges::range_difference_t<
                           std::ranges::subrange<forward_iterator<int*>, sentinel_wrapper<forward_iterator<int*>>>>>);
    }

    // bidirectional_iterator
    {
      static_assert(
          std::same_as<typename std::ranges::iterator_t< std::ranges::chunk_view<
                           std::ranges::subrange<bidirectional_iterator<int*>,
                                                 sentinel_wrapper<bidirectional_iterator<int*>>>>>::iterator_category,
                       std::input_iterator_tag>);
      static_assert(
          std::same_as<typename std::ranges::iterator_t< std::ranges::chunk_view<
                           std::ranges::subrange<bidirectional_iterator<int*>,
                                                 sentinel_wrapper<bidirectional_iterator<int*>>>>>::iterator_concept,
                       std::bidirectional_iterator_tag>);
      static_assert(
          std::same_as< typename std::ranges::iterator_t< std::ranges::chunk_view<
                            std::ranges::subrange<bidirectional_iterator<int*>,
                                                  sentinel_wrapper<bidirectional_iterator<int*>>>>>::value_type,
                        std::ranges::range_value_t<std::ranges::chunk_view<
                            std::ranges::subrange<bidirectional_iterator<int*>,
                                                  sentinel_wrapper<bidirectional_iterator<int*>>>>>>);
      static_assert(
          std::same_as<
              typename std::ranges::iterator_t< std::ranges::chunk_view<
                  std::ranges::subrange<bidirectional_iterator<int*>,
                                        sentinel_wrapper<bidirectional_iterator<int*>>>>>::difference_type,
              std::ranges::range_difference_t< std::ranges::subrange<bidirectional_iterator<int*>,
                                                                     sentinel_wrapper<bidirectional_iterator<int*>>>>>);
    }

    // random_access_iterator
    {
      static_assert(
          std::same_as<typename std::ranges::iterator_t< std::ranges::chunk_view<
                           std::ranges::subrange<random_access_iterator<int*>,
                                                 sentinel_wrapper<random_access_iterator<int*>>>>>::iterator_category,
                       std::input_iterator_tag>);
      static_assert(
          std::same_as<typename std::ranges::iterator_t< std::ranges::chunk_view<
                           std::ranges::subrange<random_access_iterator<int*>,
                                                 sentinel_wrapper<random_access_iterator<int*>>>>>::iterator_concept,
                       std::random_access_iterator_tag>);
      static_assert(
          std::same_as< typename std::ranges::iterator_t< std::ranges::chunk_view<
                            std::ranges::subrange<random_access_iterator<int*>,
                                                  sentinel_wrapper<random_access_iterator<int*>>>>>::value_type,
                        std::ranges::range_value_t<std::ranges::chunk_view<
                            std::ranges::subrange<random_access_iterator<int*>,
                                                  sentinel_wrapper<random_access_iterator<int*>>>>>>);
      static_assert(
          std::same_as<
              typename std::ranges::iterator_t< std::ranges::chunk_view<
                  std::ranges::subrange<random_access_iterator<int*>,
                                        sentinel_wrapper<random_access_iterator<int*>>>>>::difference_type,
              std::ranges::range_difference_t< std::ranges::subrange<random_access_iterator<int*>,
                                                                     sentinel_wrapper<random_access_iterator<int*>>>>>);
    }

    // contiguous_iterator
    {
      static_assert(
          std::same_as<typename std::ranges::iterator_t< std::ranges::chunk_view<
                           std::ranges::subrange<contiguous_iterator<int*>,
                                                 sentinel_wrapper<contiguous_iterator<int*>>>>>::iterator_category,
                       std::input_iterator_tag>);
      static_assert(
          std::same_as<typename std::ranges::iterator_t< std::ranges::chunk_view<
                           std::ranges::subrange<contiguous_iterator<int*>,
                                                 sentinel_wrapper<contiguous_iterator<int*>>>>>::iterator_concept,
                       std::random_access_iterator_tag>);
      static_assert(
          std::same_as<
              typename std::ranges::iterator_t< std::ranges::chunk_view<
                  std::ranges::subrange<contiguous_iterator<int*>, sentinel_wrapper<contiguous_iterator<int*>>>>>::
                  value_type,
              std::ranges::range_value_t<std::ranges::chunk_view<
                  std::ranges::subrange<contiguous_iterator<int*>, sentinel_wrapper<contiguous_iterator<int*>>>>>>);
      static_assert(
          std::same_as<
              typename std::ranges::iterator_t< std::ranges::chunk_view<
                  std::ranges::subrange<contiguous_iterator<int*>, sentinel_wrapper<contiguous_iterator<int*>>>>>::
                  difference_type,
              std::ranges::range_difference_t<
                  std::ranges::subrange<contiguous_iterator<int*>, sentinel_wrapper<contiguous_iterator<int*>>>>>);
    }

    // int*
    {
      static_assert(std::same_as<typename std::ranges::iterator_t<std::ranges::chunk_view<
                                     std::ranges::subrange<int*, sentinel_wrapper<int*>>>>::iterator_category,
                                 std::input_iterator_tag>);
      static_assert(std::same_as<typename std::ranges::iterator_t<std::ranges::chunk_view<
                                     std::ranges::subrange<int*, sentinel_wrapper<int*>>>>::iterator_concept,
                                 std::random_access_iterator_tag>);
      static_assert(
          std::same_as< typename std::ranges::iterator_t<
                            std::ranges::chunk_view<std::ranges::subrange<int*, sentinel_wrapper<int*>>>>::value_type,
                        std::ranges::range_value_t<
                            std::ranges::chunk_view<std::ranges::subrange<int*, sentinel_wrapper<int*>>>>>);
      static_assert(std::same_as<typename std::ranges::iterator_t< std::ranges::chunk_view<
                                     std::ranges::subrange<int*, sentinel_wrapper<int*>>>>::difference_type,
                                 std::ranges::range_difference_t<std::ranges::subrange<int*, sentinel_wrapper<int*>>>>);
    }
  }
}
