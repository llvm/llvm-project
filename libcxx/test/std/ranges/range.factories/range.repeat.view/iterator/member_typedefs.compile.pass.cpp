//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Test iterator category and iterator concepts.

// using index-type = conditional_t<same_as<Bound, unreachable_sentinel_t>, ptrdiff_t, Bound>;
// using iterator_concept = random_access_iterator_tag;
// using iterator_category = random_access_iterator_tag;
// using value_type = T;
// using difference_type = see below:
// If is-signed-integer-like<index-type> is true, the member typedef-name difference_type denotes
// index-type. Otherwise, it denotes IOTA-DIFF-T(index-type).

#include <cassert>
#include <concepts>
#include <cstdint>
#include <ranges>
#include <type_traits>

constexpr bool test() {
  // unbound
  {
    using Iter = std::ranges::iterator_t<std::ranges::repeat_view<int>>;
    static_assert(std::same_as<Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::same_as<Iter::iterator_category, std::random_access_iterator_tag>);
    static_assert(std::same_as<Iter::value_type, int>);
    static_assert(std::same_as<Iter::difference_type, ptrdiff_t>);
    static_assert(std::is_signed_v<Iter::difference_type>);
  }

  // bound
  {
    {
      using Iter = std::ranges::iterator_t<const std::ranges::repeat_view<int, std::int8_t>>;
      static_assert(std::same_as<Iter::iterator_concept, std::random_access_iterator_tag>);
      static_assert(std::same_as<Iter::iterator_category, std::random_access_iterator_tag>);
      static_assert(std::same_as<Iter::value_type, int>);
      static_assert(std::is_signed_v<Iter::difference_type>);
      static_assert(sizeof(Iter::difference_type) == sizeof(std::int8_t));
    }

    {
      using Iter = std::ranges::iterator_t<const std::ranges::repeat_view<int, std::uint8_t>>;
      static_assert(std::same_as<Iter::iterator_concept, std::random_access_iterator_tag>);
      static_assert(std::same_as<Iter::iterator_category, std::random_access_iterator_tag>);
      static_assert(std::same_as<Iter::value_type, int>);
      static_assert(std::is_signed_v<Iter::difference_type>);
      static_assert(sizeof(Iter::difference_type) > sizeof(std::uint8_t));
    }

    {
      using Iter = std::ranges::iterator_t<const std::ranges::repeat_view<int, std::int16_t>>;
      static_assert(std::same_as<Iter::iterator_concept, std::random_access_iterator_tag>);
      static_assert(std::same_as<Iter::iterator_category, std::random_access_iterator_tag>);
      static_assert(std::same_as<Iter::value_type, int>);
      static_assert(std::is_signed_v<Iter::difference_type>);
      static_assert(sizeof(Iter::difference_type) == sizeof(std::int16_t));
    }

    {
      using Iter = std::ranges::iterator_t<const std::ranges::repeat_view<int, std::uint16_t>>;
      static_assert(std::same_as<Iter::iterator_concept, std::random_access_iterator_tag>);
      static_assert(std::same_as<Iter::iterator_category, std::random_access_iterator_tag>);
      static_assert(std::same_as<Iter::value_type, int>);
      static_assert(std::is_signed_v<Iter::difference_type>);
      static_assert(sizeof(Iter::difference_type) > sizeof(std::uint16_t));
    }

    {
      using Iter = std::ranges::iterator_t<const std::ranges::repeat_view<int, std::int32_t>>;
      static_assert(std::same_as<Iter::iterator_concept, std::random_access_iterator_tag>);
      static_assert(std::same_as<Iter::iterator_category, std::random_access_iterator_tag>);
      static_assert(std::same_as<Iter::value_type, int>);
      static_assert(std::is_signed_v<Iter::difference_type>);
      static_assert(sizeof(Iter::difference_type) == sizeof(std::int32_t));
    }

    {
      using Iter = std::ranges::iterator_t<const std::ranges::repeat_view<int, std::uint32_t>>;
      static_assert(std::same_as<Iter::iterator_concept, std::random_access_iterator_tag>);
      static_assert(std::same_as<Iter::iterator_category, std::random_access_iterator_tag>);
      static_assert(std::same_as<Iter::value_type, int>);
      static_assert(std::is_signed_v<Iter::difference_type>);
      static_assert(sizeof(Iter::difference_type) > sizeof(std::uint32_t));
    }

    {
      using Iter = std::ranges::iterator_t<const std::ranges::repeat_view<int, std::int64_t>>;
      static_assert(std::same_as<Iter::iterator_concept, std::random_access_iterator_tag>);
      static_assert(std::same_as<Iter::iterator_category, std::random_access_iterator_tag>);
      static_assert(std::same_as<Iter::value_type, int>);
      static_assert(std::is_signed_v<Iter::difference_type>);
      static_assert(sizeof(Iter::difference_type) == sizeof(std::int64_t));
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
