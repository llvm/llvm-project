//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class T>
// using iter_reference_t = decltype(*declval<T&>());

#include <iterator>

#include <concepts>

#include "test_iterators.h"

static_assert(std::same_as<std::iter_reference_t<cpp17_input_iterator<int*> >, int&>);
static_assert(std::same_as<std::iter_reference_t<forward_iterator<int*> >, int&>);
static_assert(std::same_as<std::iter_reference_t<bidirectional_iterator<int*> >, int&>);
static_assert(std::same_as<std::iter_reference_t<random_access_iterator<int*> >, int&>);
