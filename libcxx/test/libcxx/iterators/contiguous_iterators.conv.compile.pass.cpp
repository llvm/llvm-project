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
// __static_bounded_iter<_Iter>
// __wrap_iter<_Iter>

// Verify that libc++-wrapped iterators do not permit slicing conversion or construction.

#include <array>
#include <span>
#include <type_traits>
#include <vector>

#include "test_macros.h"

struct Base {};
struct Derived : Base {};

template <class B, class D, bool = std::is_pointer<typename std::array<B, 1>::iterator>::value>
struct test_array_helper : std::true_type {
  typedef typename std::array<B, 1>::iterator BaseIter;
  typedef typename std::array<D, 1>::iterator DerivedIter;
  typedef typename std::array<B, 1>::const_iterator BaseConstIter;
  typedef typename std::array<D, 1>::const_iterator DerivedConstIter;

  static_assert(!std::is_convertible<DerivedIter, BaseIter>::value, "");
  static_assert(!std::is_convertible<DerivedIter, BaseConstIter>::value, "");
  static_assert(!std::is_convertible<DerivedConstIter, BaseConstIter>::value, "");
  static_assert(!std::is_constructible<BaseIter, DerivedIter>::value, "");
  static_assert(!std::is_constructible<BaseConstIter, DerivedIter>::value, "");
  static_assert(!std::is_constructible<BaseConstIter, DerivedConstIter>::value, "");
};

template <class B, class D>
struct test_array_helper<B, D, true> : std::true_type {};

static_assert(test_array_helper<Base, Derived>::value, "");

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
