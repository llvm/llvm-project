//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ADJACENT_HELPERS_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ADJACENT_HELPERS_H

#include <tuple>

// We intentionally don't use metaprogramming to define the expected tuple types
// because otherwise the test code would be basically the same as the source code
// we're trying to test.

template <std::size_t N, class T>
struct ExpectedTupleType;

template <class T>
struct ExpectedTupleType<1, T> {
  using type = std::tuple<T>;
};
template <class T>
struct ExpectedTupleType<2, T> {
  using type = std::tuple<T, T>;
};
template <class T>
struct ExpectedTupleType<3, T> {
  using type = std::tuple<T, T, T>;
};
template <class T>
struct ExpectedTupleType<4, T> {
  using type = std::tuple<T, T, T, T>;
};
template <class T>
struct ExpectedTupleType<5, T> {
  using type = std::tuple<T, T, T, T, T>;
};

template <std::size_t N, class T>
using expectedTupleType = typename ExpectedTupleType<N, T>::type;

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ADJACENT_HELPERS_H
