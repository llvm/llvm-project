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
// XFAIL: msvc && clang

// iterator, reverse_iterator

#include <span>

#include <iterator>

using iterator = std::span<int>::iterator;
using value_type = iterator::value_type;

static_assert(std::indirectly_readable<iterator>);
static_assert(std::indirectly_writable<iterator, value_type>);
static_assert(std::incrementable<iterator>);
