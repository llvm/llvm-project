//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class I>
// using iter_rvalue_reference;

#include <iterator>

static_assert(std::same_as<std::iter_rvalue_reference_t<int*>, int&&>);
static_assert(std::same_as<std::iter_rvalue_reference_t<const int*>, const int&&>);

void test_undefined_internal() {
  struct A {
    int& operator*() const;
  };
  static_assert(std::same_as<std::iter_rvalue_reference_t<A>, int&&>);
}
