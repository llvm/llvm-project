//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// template<class InputIterator1, class InputIterator2, class Cmp>
//     constexpr auto
//     lexicographical_compare_three_way(InputIterator1 first1, InputIterator1 last1,
//                                       InputIterator2 first2, InputIterator2 last2,
//                                       Cmp comp)
//       -> decltype(comp(*b1, *b2));

#include <array>
#include <algorithm>
#include <cassert>
#include <compare>

#include "test_macros.h"
#include "almost_satisfies_types.h"

constexpr bool incorrect_comparator(int a, int b) { return a < b; }

auto test_incorrect_comparator() {
  std::array a{90, 81};
  std::array b{10, 11};
  // expected-error-re@*:* {{static assertion failed{{.*}}The comparator passed to lexicographical_compare_three_way must return a comparison category type}}
  // expected-error@*:* {{no viable conversion}}
  return std::lexicographical_compare_three_way(a.begin(), a.end(), b.begin(), b.end(), incorrect_comparator);
}

auto test_invalid_difference_type_first(
    RandomAccessIteratorBadDifferenceType a, RandomAccessIteratorBadDifferenceType b, int* c, int* d) {
  // expected-error-re@*:* {{static assertion failed{{.*}}Using a non-integral difference_type is undefined behavior}}}}
  return std::lexicographical_compare_three_way(a, b, c, d, std::compare_three_way());
}

auto test_invalid_difference_type_second(
    int* a, int* b, RandomAccessIteratorBadDifferenceType c, RandomAccessIteratorBadDifferenceType d) {
  // expected-error-re@*:* {{static assertion failed{{.*}}Using a non-integral difference_type is undefined behavior}}}}
  return std::lexicographical_compare_three_way(a, b, c, d, std::compare_three_way());
}
