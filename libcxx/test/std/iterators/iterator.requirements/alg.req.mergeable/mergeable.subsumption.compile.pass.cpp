//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<class I1, class I2, class Out,
//     class R = ranges::less, class P1 = identity, class P2 = identity>
//   concept mergeable = see below;                           // since C++20

#include <iterator>

#include "test_macros.h"

template <class I1, class I2, class O>
void test_subsumption() requires std::input_iterator<I1> && std::input_iterator<I2>;

template <class I1, class I2, class O>
void test_subsumption() requires std::weakly_incrementable<O>;

template <class I1, class I2, class O>
void test_subsumption() requires std::indirectly_copyable<I1, O> && std::indirectly_copyable<I2, O>;

template <class I1, class I2, class O>
void test_subsumption() requires std::indirect_strict_weak_order<I1, I2>;

template <class I1, class I2, class O>
constexpr bool test_subsumption() requires std::mergeable<I1, I2, O> {
  return true;
}

static_assert(test_subsumption<int*, int*, int*>());
