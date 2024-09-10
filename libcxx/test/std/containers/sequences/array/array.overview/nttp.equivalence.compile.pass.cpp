//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <array>

// LWG-3382 NTTP for pair and array:
// Two values a1 and a2 of type array<T, N> are template-argument-equivalent if and only if each pair of corresponding
// elements in a1 and a2 are template-argument-equivalent.

#include <array>

#include <type_traits>

namespace test_full_type {
template <class T, std::size_t S, std::array<T, S> A>
struct test : std::false_type {};

template <>
struct test<int, 3, std::array<int, 3>{1, 2, 3}> : std::true_type {};

static_assert(!test<int*, 4, std::array<int*, 4>{}>::value);
static_assert(!test<int*, 3, std::array<int*, 3>{}>::value);
static_assert(!test<int, 3, std::array<int, 3>{}>::value);
static_assert(!test<int, 3, std::array<int, 3>{1}>::value);
static_assert(!test<int, 3, std::array<int, 3>{1, 2}>::value);
static_assert(!test<long, 3, std::array<long, 3>{1, 2, 3}>::value);
static_assert(!test<unsigned int, 3, std::array<unsigned int, 3>{1, 2, 3}>::value);
static_assert(test<int, 3, std::array<int, 3>{1, 2, 3}>::value);
} // namespace test_full_type

namespace test_ctad {
template <std::array A>
struct test : std::false_type {};

template <>
struct test<std::array<int, 3>{4, 5, 6}> : std::true_type {};

static_assert(!test<std::array<int*, 4>{}>::value);
static_assert(!test<std::array<int*, 3>{}>::value);
static_assert(!test<std::array<int, 3>{}>::value);
static_assert(!test<std::array<int, 3>{4}>::value);
static_assert(!test<std::array<int, 3>{4, 5}>::value);
static_assert(!test<std::array<long, 3>{4, 5, 6}>::value);
static_assert(!test<std::array<unsigned int, 3>{4, 5, 6}>::value);
static_assert(test<std::array<int, 3>{4, 5, 6}>::value);
} // namespace test_ctad

namespace test_auto {
template <auto A>
struct test : std::false_type {};

template <>
struct test<std::array<int, 3>{7, 8, 9}> : std::true_type {};

static_assert(!test<std::array<int*, 4>{}>::value);
static_assert(!test<std::array<int*, 3>{}>::value);
static_assert(!test<std::array<int, 3>{}>::value);
static_assert(!test<std::array<int, 3>{7}>::value);
static_assert(!test<std::array<int, 3>{7, 8}>::value);
static_assert(!test<std::array<long, 3>{7, 8, 9}>::value);
static_assert(!test<std::array<unsigned int, 3>{7, 8, 9}>::value);
static_assert(test<std::array<int, 3>{7, 8, 9}>::value);
} // namespace test_auto
