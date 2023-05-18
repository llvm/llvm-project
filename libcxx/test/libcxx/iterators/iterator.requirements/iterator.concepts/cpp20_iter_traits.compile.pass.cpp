//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// ITER_TRAITS(I)

// For a type I, let ITER_TRAITS(I) denote the type I if iterator_traits<I> names
// a specialization generated from the primary template. Otherwise,
// ITER_TRAITS(I) denotes iterator_traits<I>.

#include <iterator>
#include <type_traits>

#include "test_iterators.h"

struct A : random_access_iterator<int*> {};
struct B : random_access_iterator<int*> {};
struct C : random_access_iterator<int*> {};
struct D : random_access_iterator<int*> {};
template<> struct std::iterator_traits<B> {};
template<> struct std::iterator_traits<C> : std::iterator_traits<A> {};
template<> struct std::iterator_traits<D> : std::iterator_traits<int*> {};

static_assert(std::is_same<std::_ITER_TRAITS<int*>, std::iterator_traits<int*>>::value, "");
static_assert(std::is_same<std::_ITER_TRAITS<A>, A>::value, "");
static_assert(std::is_same<std::_ITER_TRAITS<B>, std::iterator_traits<B>>::value, "");
static_assert(std::is_same<std::_ITER_TRAITS<C>, std::iterator_traits<C>>::value, "");
static_assert(std::is_same<std::_ITER_TRAITS<D>, std::iterator_traits<D>>::value, "");
