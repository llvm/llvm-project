//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ADJACENT_TRANSFORM_HELPERS_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ADJACENT_TRANSFORM_HELPERS_H

#include <cassert>
#include <functional>
#include <numeric>
#include <tuple>

// intentionally not using meta programming for the expected tuple types

struct Multiply {
  template <class... Ts>
  constexpr auto operator()(Ts... args) const {
    return (1 * ... * args);
  }
};

struct MakeTuple {
  constexpr auto operator()(auto&&... args) const { return std::make_tuple(std::forward<decltype(args)>(args)...); }
};

struct Tie {
  constexpr auto operator()(auto&&... args) const { return std::tie(std::forward<decltype(args)>(args)...); }
};

struct GetFirst {
  constexpr decltype(auto) operator()(auto&& first, auto&&...) const { return std::forward<decltype(first)>(first); }
};

template <std::size_t N>
struct ValidateTieFromIndex {
  constexpr void operator()(auto&& buffer, auto&& tuple, std::size_t idx) const {
    assert(&std::get<0>(tuple) == &buffer[idx]);
    if constexpr (N >= 2)
      assert(&std::get<1>(tuple) == &buffer[idx + 1]);
    if constexpr (N >= 3)
      assert(&std::get<2>(tuple) == &buffer[idx + 2]);
    if constexpr (N >= 4)
      assert(&std::get<3>(tuple) == &buffer[idx + 3]);
    if constexpr (N >= 5)
      assert(&std::get<4>(tuple) == &buffer[idx + 4]);
  }
};

template <std::size_t N>
struct ValidateTupleFromIndex {
  constexpr void operator()(auto&& buffer, auto&& tuple, std::size_t idx) const {
    assert(std::get<0>(tuple) == buffer[idx]);
    if constexpr (N >= 2)
      assert(std::get<1>(tuple) == buffer[idx + 1]);
    if constexpr (N >= 3)
      assert(std::get<2>(tuple) == buffer[idx + 2]);
    if constexpr (N >= 4)
      assert(std::get<3>(tuple) == buffer[idx + 3]);
    if constexpr (N >= 5)
      assert(std::get<4>(tuple) == buffer[idx + 4]);
  }
};

template <std::size_t N>
struct ValidateGetFirstFromIndex {
  constexpr void operator()(auto&& buffer, auto&& result, std::size_t idx) const { assert(&result == &buffer[idx]); }
};

template <std::size_t N>
struct ValidateMultiplyFromIndex {
  constexpr void operator()(auto&& buffer, auto&& result, std::size_t idx) const {
    auto expected = std::accumulate(buffer + idx, buffer + idx + N, 1, std::multiplies<>());
    assert(result == expected);
  }
};

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

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ADJACENT_TRANSFORM_HELPERS_H
