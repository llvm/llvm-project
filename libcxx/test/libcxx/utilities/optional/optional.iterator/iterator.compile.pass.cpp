//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <optional>

// UNSUPPORTED: libcpp-has-abi-bounded-iterators-in-optional

// template <class T> class optional::iterator;
// template <class T> class optional::const_iterator;

#include <optional>
#include <type_traits>
#include <vector>

template <typename T>
concept has_iterator_aliases = requires {
  typename T::iterator;
  typename T::const_iterator;
};

static_assert(has_iterator_aliases<std::optional<int>>);
static_assert(has_iterator_aliases<std::optional<const int>>);
static_assert(has_iterator_aliases<std::optional<int&>>);
static_assert(has_iterator_aliases<std::optional<const int&>>);
static_assert(!has_iterator_aliases<std::optional<int (&)[1]>>);
static_assert(!has_iterator_aliases<std::optional<int (&)()>>);

using Iter1  = std::optional<int>::iterator;
using Iter2  = std::optional<double>::iterator;
using Iter3  = std::optional<int>::const_iterator;
using Iter4  = std::optional<double>::const_iterator;
using VIter1 = std::vector<int>::iterator;
using VIter2 = std::vector<int>::const_iterator;

static_assert(std::is_convertible_v<Iter1, Iter1>);
static_assert(!std::is_convertible_v<Iter1, Iter2>);
static_assert(std::is_convertible_v<Iter1, Iter3>);
static_assert(!std::is_convertible_v<Iter1, Iter4>);
static_assert(!std::is_convertible_v<Iter1, VIter1>);
static_assert(!std::is_convertible_v<Iter1, VIter2>);

static_assert(std::is_convertible_v<Iter2, Iter2>);
static_assert(!std::is_convertible_v<Iter2, Iter1>);
static_assert(!std::is_convertible_v<Iter2, Iter3>);
static_assert(std::is_convertible_v<Iter2, Iter4>);
static_assert(!std::is_convertible_v<Iter2, VIter1>);
static_assert(!std::is_convertible_v<Iter2, VIter2>);

static_assert(std::is_convertible_v<Iter3, Iter3>);
static_assert(!std::is_convertible_v<Iter3, Iter1>);
static_assert(!std::is_convertible_v<Iter3, Iter2>);
static_assert(!std::is_convertible_v<Iter3, Iter4>);
static_assert(!std::is_convertible_v<Iter3, VIter1>);
static_assert(!std::is_convertible_v<Iter3, VIter2>);

static_assert(std::is_convertible_v<Iter4, Iter4>);
static_assert(!std::is_convertible_v<Iter4, Iter1>);
static_assert(!std::is_convertible_v<Iter4, Iter2>);
static_assert(!std::is_convertible_v<Iter4, Iter3>);
static_assert(!std::is_convertible_v<Iter4, VIter1>);
static_assert(!std::is_convertible_v<Iter4, VIter2>);
