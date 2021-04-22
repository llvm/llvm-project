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

// ostreambuf_iterator

#include <iterator>

#include <ostream>
#include <string>

using iterator = std::ostreambuf_iterator<char>;
static_assert(!std::indirectly_readable<iterator>);
static_assert(std::indirectly_writable<iterator, char>);
