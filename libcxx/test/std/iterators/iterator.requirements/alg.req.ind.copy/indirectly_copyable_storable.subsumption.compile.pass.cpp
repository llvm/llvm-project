//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class In, class Out>
// concept indirectly_copyable_storable;

#include <iterator>

template<class I, class O>
  requires std::indirectly_copyable<I, O>
constexpr bool indirectly_copyable_storable_subsumption() {
  return false;
}

template<class I, class O>
  requires std::indirectly_copyable_storable<I, O>
constexpr bool indirectly_copyable_storable_subsumption() {
  return true;
}

static_assert(indirectly_copyable_storable_subsumption<int*, int*>());
