//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// test that iterators from different types of flat_set are not compatible

#include <deque>
#include <functional>
#include <flat_set>
#include <type_traits>

using Iter1 = std::flat_set<int>::iterator;
using Iter2 = std::flat_set<double>::iterator;
using Iter3 = std::flat_set<int, std::greater<>>::iterator;
using Iter4 = std::flat_set<int, std::less<int>, std::deque<int>>::iterator;

static_assert(std::is_convertible_v<Iter1, Iter1>);
static_assert(!std::is_convertible_v<Iter1, Iter2>);
static_assert(!std::is_convertible_v<Iter1, Iter3>);
static_assert(!std::is_convertible_v<Iter1, Iter4>);

static_assert(!std::is_convertible_v<Iter2, Iter1>);
static_assert(std::is_convertible_v<Iter2, Iter2>);
static_assert(!std::is_convertible_v<Iter2, Iter3>);
static_assert(!std::is_convertible_v<Iter2, Iter4>);

static_assert(!std::is_convertible_v<Iter3, Iter1>);
static_assert(!std::is_convertible_v<Iter3, Iter2>);
static_assert(std::is_convertible_v<Iter3, Iter3>);
static_assert(!std::is_convertible_v<Iter3, Iter4>);

static_assert(!std::is_convertible_v<Iter4, Iter1>);
static_assert(!std::is_convertible_v<Iter4, Iter2>);
static_assert(!std::is_convertible_v<Iter4, Iter3>);
static_assert(std::is_convertible_v<Iter4, Iter4>);
