//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class T>
//     concept default_initializable = constructible_from<T> &&
//     requires { T{}; } &&
//     is-default-initializable<T>;

#include <concepts>
#include <cassert>

#include "test_macros.h"

template<class T>
concept brace_initializable = requires { T{}; };

void test() {
    // LWG3149
    // Changed the concept from constructible_from<T>
    // to constructible_from<T> &&
    //    requires { T{}; } && is-default-initializable <T>
    struct S0 { explicit S0() = default; };
    S0 x0;
    S0 y0{};
    static_assert(std::constructible_from<S0>);
    static_assert(brace_initializable<S0>);
    LIBCPP_STATIC_ASSERT(std::__default_initializable<S0>);
    static_assert(std::default_initializable<S0>);

    struct S1 { S0 x; }; // Note: aggregate
    S1 x1;
    S1 y1{}; // expected-error {{chosen constructor is explicit in copy-initialization}}
    static_assert(std::constructible_from<S1>);
    static_assert(!brace_initializable<S1>);
    LIBCPP_STATIC_ASSERT(std::__default_initializable<S1>);
    static_assert(!std::default_initializable<S1>);

    const int x2; // expected-error {{default initialization of an object of const type 'const int'}}
    const int y2{};

    static_assert(std::constructible_from<const int>);
    static_assert(brace_initializable<const int>);
    LIBCPP_STATIC_ASSERT(!std::__default_initializable<const int>);
    static_assert(!std::default_initializable<const int>);

    const int x3[1]; // expected-error {{default initialization of an object of const type 'const int[1]'}}
    const int y3[1]{};
    static_assert(std::constructible_from<const int[1]>);
    static_assert(brace_initializable<const int[1]>);
    LIBCPP_STATIC_ASSERT(!std::__default_initializable<const int[1]>);
    static_assert(!std::default_initializable<const int[1]>);

    // Zero-length array extension
    const int x4[]; // expected-error {{definition of variable with array type needs an explicit size or an initializer}}
    const int y4[]{};
    static_assert(!std::constructible_from<const int[]>);
    static_assert(brace_initializable<const int[]>);
    LIBCPP_STATIC_ASSERT(!std::__default_initializable<const int[]>);
    static_assert(!std::default_initializable<const int[]>);
}

int main(int, char**) {
    test();

    return 0;
}
