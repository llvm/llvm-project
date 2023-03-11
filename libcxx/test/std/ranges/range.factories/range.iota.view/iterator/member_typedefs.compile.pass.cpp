//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Test iterator category and iterator concepts.

#include <cassert>
#include <cstdint>
#include <ranges>
#include <type_traits>

#include "test_macros.h"
#include "../types.h"

struct Decrementable {
  using difference_type = int;

  auto operator<=>(const Decrementable&) const = default;

  constexpr Decrementable& operator++();
  constexpr Decrementable  operator++(int);
  constexpr Decrementable& operator--();
  constexpr Decrementable  operator--(int);
};

struct Incrementable {
  using difference_type = int;

  auto operator<=>(const Incrementable&) const = default;

  constexpr Incrementable& operator++();
  constexpr Incrementable  operator++(int);
};

struct BigType {
  char buffer[128];

  using difference_type = int;

  auto operator<=>(const BigType&) const = default;

  constexpr BigType& operator++();
  constexpr BigType  operator++(int);
};

struct CharDifferenceType {
  using difference_type = signed char;

  auto operator<=>(const CharDifferenceType&) const = default;

  constexpr CharDifferenceType& operator++();
  constexpr CharDifferenceType  operator++(int);
};

template<class T>
concept HasIteratorCategory = requires { typename std::ranges::iterator_t<T>::iterator_category; };

void test() {
  {
    const std::ranges::iota_view<char> io(0);
    using Iter = decltype(io.begin());
    static_assert(std::same_as<Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::same_as<Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::same_as<Iter::value_type, char>);
    static_assert(sizeof(Iter::difference_type) > sizeof(char));
    static_assert(std::is_signed_v<Iter::difference_type>);
    LIBCPP_STATIC_ASSERT(std::same_as<Iter::difference_type, int>);
  }
  {
    const std::ranges::iota_view<short> io(0);
    using Iter = decltype(io.begin());
    static_assert(std::same_as<Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::same_as<Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::same_as<Iter::value_type, short>);
    static_assert(sizeof(Iter::difference_type) > sizeof(short));
    static_assert(std::is_signed_v<Iter::difference_type>);
    LIBCPP_STATIC_ASSERT(std::same_as<Iter::difference_type, int>);
  }
  {
    const std::ranges::iota_view<int> io(0);
    using Iter = decltype(io.begin());
    static_assert(std::same_as<Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::same_as<Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::same_as<Iter::value_type, int>);
    static_assert(sizeof(Iter::difference_type) > sizeof(int));
    static_assert(std::is_signed_v<Iter::difference_type>);
    // If we're compiling for 32 bit or windows, int and long are the same size, so long long is the correct difference type.
#if INTPTR_MAX == INT32_MAX || defined(_WIN32)
    LIBCPP_STATIC_ASSERT(std::same_as<Iter::difference_type, long long>);
#else
    LIBCPP_STATIC_ASSERT(std::same_as<Iter::difference_type, long>);
#endif
  }
  {
    const std::ranges::iota_view<long> io(0);
    using Iter = decltype(io.begin());
    static_assert(std::same_as<Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::same_as<Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::same_as<Iter::value_type, long>);
    // Same as below, if there is no type larger than long, we can just use that.
    static_assert(sizeof(Iter::difference_type) >= sizeof(long));
    static_assert(std::is_signed_v<Iter::difference_type>);
    LIBCPP_STATIC_ASSERT(std::same_as<Iter::difference_type, long long>);
  }
  {
    const std::ranges::iota_view<long long> io(0);
    using Iter = decltype(io.begin());
    static_assert(std::same_as<Iter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::same_as<Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::same_as<Iter::value_type, long long>);
    // No integer is larger than long long, so it is OK to use long long as the difference type here:
    // https://eel.is/c++draft/range.iota.view#1.3
    static_assert(sizeof(Iter::difference_type) >= sizeof(long long));
    static_assert(std::is_signed_v<Iter::difference_type>);
    LIBCPP_STATIC_ASSERT(std::same_as<Iter::difference_type, long long>);
  }
  {
    const std::ranges::iota_view<Decrementable> io;
    using Iter = decltype(io.begin());
    static_assert(std::same_as<Iter::iterator_concept, std::bidirectional_iterator_tag>);
    static_assert(std::same_as<Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::same_as<Iter::value_type, Decrementable>);
    static_assert(std::same_as<Iter::difference_type, int>);
  }
  {
    const std::ranges::iota_view<Incrementable> io;
    using Iter = decltype(io.begin());
    static_assert(std::same_as<Iter::iterator_concept, std::forward_iterator_tag>);
    static_assert(std::same_as<Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::same_as<Iter::value_type, Incrementable>);
    static_assert(std::same_as<Iter::difference_type, int>);
  }
  {
    const std::ranges::iota_view<NotIncrementable> io(NotIncrementable(0));
    using Iter = decltype(io.begin());
    static_assert(std::same_as<Iter::iterator_concept, std::input_iterator_tag>);
    static_assert(!HasIteratorCategory<std::ranges::iota_view<NotIncrementable>>);
    static_assert(std::same_as<Iter::value_type, NotIncrementable>);
    static_assert(std::same_as<Iter::difference_type, int>);
  }
  {
    const std::ranges::iota_view<BigType> io;
    using Iter = decltype(io.begin());
    static_assert(std::same_as<Iter::iterator_concept, std::forward_iterator_tag>);
    static_assert(std::same_as<Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::same_as<Iter::value_type, BigType>);
    static_assert(std::same_as<Iter::difference_type, int>);
  }
  {
    const std::ranges::iota_view<CharDifferenceType> io;
    using Iter = decltype(io.begin());
    static_assert(std::same_as<Iter::iterator_concept, std::forward_iterator_tag>);
    static_assert(std::same_as<Iter::iterator_category, std::input_iterator_tag>);
    static_assert(std::same_as<Iter::value_type, CharDifferenceType>);
    static_assert(std::same_as<Iter::difference_type, signed char>);
  }
}
