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

// regex_token_iterator

#include <regex>

#include <iterator>

static_assert(std::indirectly_readable<std::cregex_token_iterator>);
static_assert(!std::indirectly_writable<std::cregex_token_iterator, char>);
