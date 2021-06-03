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

// projected

#include <iterator>

#include <concepts>
#include <functional>

#include "test_iterators.h"

using IntPtr = std::projected<int const*, std::identity>;
static_assert(std::same_as<IntPtr::value_type, int>);
static_assert(std::same_as<decltype(*std::declval<IntPtr>()), int const&>);
static_assert(std::same_as<std::iter_difference_t<IntPtr>, std::ptrdiff_t>);

struct S { };

using Cpp17InputIterator = std::projected<cpp17_input_iterator<S*>, int S::*>;
static_assert(std::same_as<Cpp17InputIterator::value_type, int>);
static_assert(std::same_as<decltype(*std::declval<Cpp17InputIterator>()), int&>);
static_assert(std::same_as<std::iter_difference_t<Cpp17InputIterator>, std::ptrdiff_t>);

using Cpp20InputIterator = std::projected<cpp20_input_iterator<S*>, int S::*>;
static_assert(std::same_as<Cpp20InputIterator::value_type, int>);
static_assert(std::same_as<decltype(*std::declval<Cpp20InputIterator>()), int&>);
static_assert(std::same_as<std::iter_difference_t<Cpp20InputIterator>, std::ptrdiff_t>);

using ForwardIterator = std::projected<forward_iterator<S*>, int (S::*)()>;
static_assert(std::same_as<ForwardIterator::value_type, int>);
static_assert(std::same_as<decltype(*std::declval<ForwardIterator>()), int>);
static_assert(std::same_as<std::iter_difference_t<ForwardIterator>, std::ptrdiff_t>);

using BidirectionalIterator = std::projected<bidirectional_iterator<S*>, S* (S::*)() const>;
static_assert(std::same_as<BidirectionalIterator::value_type, S*>);
static_assert(std::same_as<decltype(*std::declval<BidirectionalIterator>()), S*>);
static_assert(std::same_as<std::iter_difference_t<BidirectionalIterator>, std::ptrdiff_t>);

using RandomAccessIterator = std::projected<random_access_iterator<S*>, S && (S::*)()>;
static_assert(std::same_as<RandomAccessIterator::value_type, S>);
static_assert(std::same_as<decltype(*std::declval<RandomAccessIterator>()), S&&>);
static_assert(std::same_as<std::iter_difference_t<RandomAccessIterator>, std::ptrdiff_t>);

using ContiguousIterator = std::projected<contiguous_iterator<S*>, S& (S::*)() const>;
static_assert(std::same_as<ContiguousIterator::value_type, S>);
static_assert(std::same_as<decltype(*std::declval<ContiguousIterator>()), S&>);
static_assert(std::same_as<std::iter_difference_t<ContiguousIterator>, std::ptrdiff_t>);

template <class I, class F>
constexpr bool projectable = requires {
  typename std::projected<I, F>;
};

static_assert(!projectable<int, void (*)(int)>); // int isn't indirectly_readable
static_assert(!projectable<S, void (*)(int)>);   // S isn't weakly_incrementable
static_assert(!projectable<int*, void(int)>);    // void(int) doesn't satisfy indirectly_regular_unary_invcable
