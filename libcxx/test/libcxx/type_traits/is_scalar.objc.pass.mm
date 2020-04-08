//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03
// REQUIRES: has-fblocks
// ADDITIONAL_COMPILE_FLAGS: -fblocks

// <type_traits>

// std::is_scalar

// Make sure we report that blocks are scalar types.

#include <type_traits>

struct Foo { };
template <int> struct Arg { };

static_assert(std::is_scalar<void (^)(void)>::value, "");
static_assert(std::is_scalar<void (^)()>::value, "");
static_assert(std::is_scalar<void (^)(Arg<0>)>::value, "");
static_assert(std::is_scalar<void (^)(Arg<0>, Arg<1>)>::value, "");
static_assert(std::is_scalar<void (^)(Arg<0>, Arg<1>, Arg<2>)>::value, "");
static_assert(std::is_scalar<Foo (^)(void)>::value, "");
static_assert(std::is_scalar<Foo (^)()>::value, "");
static_assert(std::is_scalar<Foo (^)(Arg<0>)>::value, "");
static_assert(std::is_scalar<Foo (^)(Arg<0>, Arg<1>)>::value, "");
static_assert(std::is_scalar<Foo (^)(Arg<0>, Arg<1>, Arg<2>)>::value, "");


int main(int, char**) {
    return 0;
}
