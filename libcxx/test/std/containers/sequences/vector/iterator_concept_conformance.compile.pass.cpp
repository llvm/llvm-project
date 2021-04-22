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

// iterator, const_iterator, reverse_iterator, const_reverse_iterator

#include <vector>

#include <iterator>

using iterator = std::vector<int>::iterator;
using const_iterator = std::vector<int>::const_iterator;
using value_type = iterator::value_type;

static_assert(std::indirectly_readable<iterator>);
static_assert(std::indirectly_writable<iterator, value_type>);

static_assert(std::indirectly_readable<const_iterator>);
static_assert(!std::indirectly_writable<const_iterator, value_type>);
