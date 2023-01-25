//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ranges>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Check that different_from.h is self-contained
// ADDITIONAL_COMPILE_FLAGS: -Wno-private-header

#include <__concepts/different_from.h>

static_assert(std::__different_from<int, char>);
static_assert(std::__different_from<const int, char>);
static_assert(!std::__different_from<const int, int>);
static_assert(!std::__different_from<const volatile int, int>);
static_assert(!std::__different_from<const int&, int>);
static_assert(!std::__different_from<int&, int>);
static_assert(!std::__different_from<int&&, int>);
static_assert(!std::__different_from<int, int&>);
static_assert(!std::__different_from<int&, const int&>);
static_assert(!std::__different_from<int(&)(), int()>);
static_assert(std::__different_from<int(&)(), int(*)()>);
