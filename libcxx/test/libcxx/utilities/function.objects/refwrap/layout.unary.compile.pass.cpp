//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc

// ensure that unary_function always has the same ABI

#include <functional>

struct S1 : std::negate<int>, std::bit_not<int> {};

static_assert(sizeof(S1) == 2, "");

struct S2 : std::negate<int> { char c; };

static_assert(sizeof(S2) == 1, "");
