//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <cassert>
#include <cstddef>
#include <ranges>
#include <vector>

#include "test_macros.h"
#define TEST_HAS_NO_INT128 // Size cannot be larger than 64 bits
#include "type_algorithms.h"

#include "types.h"

// #include <compare>

// class IntegerLike {
//   int value;

// public:
//   // Constructor
//   IntegerLike(int v = 0) : value(v) {}

//   // Conversion to int
//   operator int() const { return value; }

//   // Conversion to std::size_t
//   // This is necessary for std::ranges::views::indices to work with IntegerLike
//   operator std::size_t() const { return static_cast<std::size_t>(value); }

//   // Equality and comparison
//   auto operator<=>(const IntegerLike&) const = default;

//   // Arithmetic
//   IntegerLike operator+(const IntegerLike& other) const { return IntegerLike(value + other.value); }

//   IntegerLike operator-(const IntegerLike& other) const { return IntegerLike(value - other.value); }

//   IntegerLike operator*(const IntegerLike& other) const { return IntegerLike(value * other.value); }

//   IntegerLike operator/(const IntegerLike& other) const { return IntegerLike(value / other.value); }

//   // Compound assignment
//   IntegerLike& operator+=(const IntegerLike& other) {
//     value += other.value;
//     return *this;
//   }

//   IntegerLike& operator-=(const IntegerLike& other) {
//     value -= other.value;
//     return *this;
//   }

//   IntegerLike& operator*=(const IntegerLike& other) {
//     value *= other.value;
//     return *this;
//   }

//   IntegerLike& operator/=(const IntegerLike& other) {
//     value /= other.value;
//     return *this;
//   }

//   // Increment / Decrement
//   IntegerLike& operator++() {
//     ++value;
//     return *this;
//   }

//   IntegerLike operator++(int) {
//     IntegerLike tmp = *this;
//     ++(*this);
//     return tmp;
//   }

//   IntegerLike& operator--() {
//     --value;
//     return *this;
//   }

//   IntegerLike operator--(int) {
//     IntegerLike tmp = *this;
//     --(*this);
//     return tmp;
//   }
// };

// Test SFINAE.

template <typename SizeType>
concept HasIndices = requires(SizeType s) { std::ranges::views::indices(s); };

struct NotIntegerLike {};

struct IntegerTypesTest {
  template <class T>
  constexpr void operator()() {
    static_assert(HasIndices<T>);
  }
};

void test_SFIANE() {
  static_assert(HasIndices<std::size_t>);
  types::for_each(types::integer_types(), IntegerTypesTest{});

  static_assert(!HasIndices<bool>);
  static_assert(!HasIndices<float>);
  static_assert(!HasIndices<void>);
  static_assert(!HasIndices<SomeInt>); // Does satisfy is_integer_like, but not the conversion to std::size_t
  static_assert(!HasIndices<NotIntegerLike>);
}

constexpr bool test() {
  {
    // Check that the indices view works as expected
    auto indices_view = std::ranges::views::indices(5);
    assert(indices_view.size() == 5);

    // This should be valid, as indices_view is a range of integers
    assert(indices_view[0] == 0);
    assert(indices_view[1] == 1);
    assert(indices_view[2] == 2);
    assert(indices_view[3] == 3);
    assert(indices_view[4] == 4);

    // Check that the view is a range
    static_assert(std::ranges::range<decltype(indices_view)>);
  }

  //   {
  //     // Check that the indices view works as expected
  //     auto indices_view = std::ranges::views::indices(IntegerLike{5});
  //     assert(indices_view.size() == 5);

  //     // Check that the view is a range
  //     static_assert(std::ranges::range<decltype(indices_view)>);

  //     // This should be valid, as indices_view is a range of integers
  //     assert(indices_view[0] == 0);
  //     assert(indices_view[1] == 1);
  //     assert(indices_view[2] == 2);
  //     assert(indices_view[3] == 3);
  //     assert(indices_view[4] == 4);
  //   }

  {
    std::vector v(5, 0);

    // Check that the indices view works as expected
    auto indices_view = std::ranges::views::indices(std::ranges::size(v));
    assert(indices_view.size() == 5);

    // Check that the view is a range
    static_assert(std::ranges::range<decltype(indices_view)>);

    // This should be valid, as indices_view is a range of integers
    assert(indices_view[0] == 0);
    assert(indices_view[1] == 1);
    assert(indices_view[2] == 2);
    assert(indices_view[3] == 3);
    assert(indices_view[4] == 4);
  }

  {
    std::vector v(5, SomeInt{});

    // Check that the indices view works as expected
    auto indices_view = std::ranges::views::indices(std::ranges::size(v));
    assert(indices_view.size() == 5);

    // Check that the view is a range
    static_assert(std::ranges::range<decltype(indices_view)>);

    // This should be valid, as indices_view is a range of integers
    assert(indices_view[0] == 0);
    assert(indices_view[1] == 1);
    assert(indices_view[2] == 2);
    assert(indices_view[3] == 3);
    assert(indices_view[4] == 4);
  }

  return true;
}

int main(int, char**) {
  test_SFIANE();

  test();
  static_assert(test());

  return 0;
}
