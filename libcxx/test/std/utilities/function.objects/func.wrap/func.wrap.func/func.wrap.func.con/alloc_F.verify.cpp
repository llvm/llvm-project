//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// UNSUPPORTED: c++03, c++11, c++14

// class function<R(ArgTypes...)>

// template<class F, class A> function(allocator_arg_t, const A&, F);
//
// This signature was removed in C++17

#include <functional>
#include <memory>

void foo(int) {}

void f() {
    std::function<void(int)> f(std::allocator_arg, std::allocator<int>(), foo); // expected-error {{no matching constructor for initialization of}}
}
