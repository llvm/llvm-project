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

// template<class I>
// using iter_rvalue_reference;

#include <iterator>

#include <concepts>
#include <list>
#include <vector>

static_assert(std::same_as<std::iter_rvalue_reference_t<std::vector<int>::iterator&>, int&&>);
static_assert(std::same_as<std::iter_rvalue_reference_t<std::vector<int>::const_iterator>, int const&&>);
static_assert(std::same_as<std::iter_rvalue_reference_t<std::list<int const>::iterator>, int const&&>);

int main(int, char**) { return 0; }
