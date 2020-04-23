//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>
// XFAIL: c++98, c++03, c++11, c++14

// class function<R(ArgTypes...)>

// template<class A> function(allocator_arg_t, const A&, nullptr_t);
//
// This signature was removed in C++17

// This test runs in C++03, but we have deprecated using std::function in C++03.
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <functional>
#include <cassert>

#include "min_allocator.h"

int main(int, char**)
{
    std::function<int(int)> f(std::allocator_arg, std::allocator<int>(), nullptr);

  return 0;
}
