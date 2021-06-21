//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10

// class std::ranges::subrange;

#include <ranges>

#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"

template<std::ranges::subrange_kind K, class... Args>
concept ValidSubrangeKind = requires { typename std::ranges::subrange<Args..., K>; };

template<class... Args>
concept ValidSubrange = requires { typename std::ranges::subrange<Args...>; };

static_assert( ValidSubrange<forward_iterator<int*>>);
static_assert( ValidSubrange<forward_iterator<int*>, forward_iterator<int*>>);
static_assert( ValidSubrangeKind<std::ranges::subrange_kind::unsized, forward_iterator<int*>, forward_iterator<int*>>);
static_assert( ValidSubrangeKind<std::ranges::subrange_kind::sized, forward_iterator<int*>, forward_iterator<int*>>);
// Wrong sentinel type.
static_assert(!ValidSubrange<forward_iterator<int*>, int*>);
static_assert( ValidSubrange<int*>);
static_assert( ValidSubrange<int*, int*>);
// Must be sized.
static_assert(!ValidSubrangeKind<std::ranges::subrange_kind::unsized, int*, int*>);
static_assert( ValidSubrangeKind<std::ranges::subrange_kind::sized, int*, int*>);
// Wrong sentinel type.
static_assert(!ValidSubrange<int*, forward_iterator<int*>>);
// Not an iterator.
static_assert(!ValidSubrange<int>);
