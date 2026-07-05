//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class T>
// concept random_access_iterator;

#include <iterator>

#include <concepts>

template<std::bidirectional_iterator I>
requires std::derived_from<std::_ITER_CONCEPT<I>, std::random_access_iterator_tag>
constexpr bool check_subsumption() {
  return false;
}

template<std::random_access_iterator>
constexpr bool check_subsumption() {
  return true;
}

static_assert(check_subsumption<int*>());
