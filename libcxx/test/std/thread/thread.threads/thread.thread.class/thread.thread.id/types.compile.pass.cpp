//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-threads

// <thread>

// template<class charT> struct formatter<thread::id, charT>;

// [format.formatter.spec]/4
//   If the library provides an explicit or partial specialization of
//   formatter<T, charT>, that specialization is enabled and meets the
//   Formatter requirements except as noted otherwise.
//
// Test parts of the BasicFormatter requirements. Like the formattable concept
// it uses the semiregular concept. This test does not use the formattable
// concept since the intent is for the formatter to be available without
// including the <format> header.

// TODO FMT Evaluate what to do with [format.formatter.spec]/2
// [format.formatter.spec]/2
//   Each header that declares the template formatter provides the following
//   enabled specializations:
// Then there is a list of formatters, but is that really useful?
// Note this should be discussed in LEWG.

#include <concepts>
#include <format>
#include <thread>

#include "test_macros.h"

static_assert(std::semiregular<std::formatter<std::thread::id, char>>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::semiregular<std::formatter<std::thread::id, wchar_t>>);
#endif
