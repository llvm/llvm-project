//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// [container.adaptors.format]/1
//   For each of queue, priority_queue, and stack, the library provides the
//   following formatter specialization where adaptor-type is the name of the
//   template:
//    template<class charT, class T, formattable<charT> Container, class... U>
//    struct formatter<adaptor-type<T, Container, U...>, charT>;
//
// Note it is unspecified in which header the adaptor formatters reside. In
// libc++ they are in <format>. However their own headers are still required for
// the declarations of these types.

// [format.formatter.spec]/4
//   If the library provides an explicit or partial specialization of
//   formatter<T, charT>, that specialization is enabled and meets the
//   Formatter requirements except as noted otherwise.
//
// Tests parts of the BasicFormatter requirements. Like the formattable concept
// it uses the semiregular concept. The test does not use the formattable
// concept to be similar to tests for formatters not provided by the <format>
// header.

#include <concepts>
#include <format>
#include <queue>
#include <stack>

#include "test_macros.h"

static_assert(std::semiregular<std::formatter<std::queue<int>, char>>);
static_assert(std::semiregular<std::formatter<std::priority_queue<int>, char>>);
static_assert(std::semiregular<std::formatter<std::stack<int>, char>>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::semiregular<std::formatter<std::queue<int>, wchar_t>>);
static_assert(std::semiregular<std::formatter<std::priority_queue<int>, wchar_t>>);
static_assert(std::semiregular<std::formatter<std::stack<int>, wchar_t>>);
#endif // TEST_HAS_NO_WIDE_CHARACTERS
