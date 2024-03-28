//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: clang-17

// <utility>

// Two values p1 and p2 of type pair<T, U> are template-argument-equivalent ([temp.type]) if and only if
// p1.first and p2.first are template-argument-equivalent and p1.second and p2.second are template-argument-equivalent.

#include <utility>

#include <type_traits>

int i = 0;
int j = 1;

namespace test_full_type {
template <class T, class U, std::pair<T, U> P>
struct test : std::false_type {};

template <>
struct test<int&, int, std::pair<int&, int>{i, 5}> : std::true_type {};

static_assert(!test<int*, int*, std::pair<int*, int*>{}>::value);
static_assert(!test<int*, int, std::pair<int*, int>{}>::value);
static_assert(!test<int&, int*, std::pair<int&, int*>{i, nullptr}>::value);
static_assert(!test<int&, int, std::pair<int&, int>{j, 0}>::value);
static_assert(!test<int&, int, std::pair<int&, int>{j, 5}>::value);
static_assert(!test<int&, int, std::pair<int&, int>{i, 0}>::value);
static_assert(!test<int&, unsigned int, std::pair<int&, unsigned int>{j, 0}>::value);
static_assert(test<int&, int, std::pair<int&, int>{i, 5}>::value);
} // namespace test_full_type

namespace test_ctad {
template <std::pair P>
struct test : std::false_type {};

template <>
struct test<std::pair<int&, int>{i, 10}> : std::true_type {};

static_assert(!test<std::pair<int*, int*>{}>::value);
static_assert(!test<std::pair<int*, int>{}>::value);
static_assert(!test<std::pair<int&, int*>{i, nullptr}>::value);
static_assert(!test<std::pair<int&, int>{j, 0}>::value);
static_assert(!test<std::pair<int&, int>{j, 10}>::value);
static_assert(!test<std::pair<int&, int>{i, 0}>::value);
static_assert(!test<std::pair<int&, unsigned int>{j, 0}>::value);
static_assert(test<std::pair<int&, int>{i, 10}>::value);
} // namespace test_ctad

namespace test_auto {
template <auto P>
struct test : std::false_type {};

template <>
struct test<std::pair<int&, int>{i, 15}> : std::true_type {};

static_assert(!test<std::pair<int*, int*>{}>::value);
static_assert(!test<std::pair<int*, int>{}>::value);
static_assert(!test<std::pair<int&, int*>{i, nullptr}>::value);
static_assert(!test<std::pair<int&, int>{j, 0}>::value);
static_assert(!test<std::pair<int&, int>{j, 15}>::value);
static_assert(!test<std::pair<int&, int>{i, 0}>::value);
static_assert(!test<std::pair<int&, unsigned int>{j, 0}>::value);
static_assert(test<std::pair<int&, int>{i, 15}>::value);
} // namespace test_auto
