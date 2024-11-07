//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// <iterator>

// __bounded_iter<_Iter>
// __wrap_iter<_Iter>

// Verify that libc++-wrapped iterators do not permit slicing conversion or construction.

#include <array>
#include <vector>
#include <span>
#include <type_traits>

#include "test_macros.h"

struct Base {};
struct Derived : Base {};

#ifdef _LIBCPP_ABI_USE_WRAP_ITER_IN_STD_ARRAY
static_assert(!std::is_convertible<std::array<Derived, 1>::iterator, std::array<Base, 1>::iterator>::value, "");
static_assert(!std::is_convertible<std::array<Derived, 1>::iterator, std::array<Base, 1>::const_iterator>::value, "");
static_assert(!std::is_convertible<std::array<Derived, 1>::const_iterator, std::array<Base, 1>::const_iterator>::value,
              "");
static_assert(!std::is_constructible<std::array<Base, 1>::iterator, std::array<Derived, 1>::iterator>::value, "");
static_assert(!std::is_constructible<std::array<Base, 1>::const_iterator, std::array<Derived, 1>::iterator>::value, "");
static_assert(
    !std::is_constructible<std::array<Base, 1>::const_iterator, std::array<Derived, 1>::const_iterator>::value, "");
#endif

static_assert(!std::is_convertible<std::vector<Derived>::iterator, std::vector<Base>::iterator>::value, "");
static_assert(!std::is_convertible<std::vector<Derived>::iterator, std::vector<Base>::const_iterator>::value, "");
static_assert(!std::is_convertible<std::vector<Derived>::const_iterator, std::vector<Base>::const_iterator>::value, "");
static_assert(!std::is_constructible<std::vector<Base>::iterator, std::vector<Derived>::iterator>::value, "");
static_assert(!std::is_constructible<std::vector<Base>::const_iterator, std::vector<Derived>::iterator>::value, "");
static_assert(!std::is_constructible<std::vector<Base>::const_iterator, std::vector<Derived>::const_iterator>::value,
              "");

#if TEST_STD_VER >= 20
static_assert(!std::is_convertible_v<std::span<Derived>::iterator, std::span<Base>::iterator>);
static_assert(!std::is_convertible_v<std::span<Derived>::iterator, std::span<const Base>::iterator>);
static_assert(!std::is_convertible_v<std::span<const Derived>::iterator, std::span<Base>::iterator>);
static_assert(!std::is_constructible_v<std::span<Base>::iterator, std::span<Derived>::iterator>);
static_assert(!std::is_constructible_v<std::span<Base>::iterator, std::span<const Derived>::iterator>);
static_assert(!std::is_constructible_v<std::span<const Base>::iterator, std::span<const Derived>::iterator>);
#endif
