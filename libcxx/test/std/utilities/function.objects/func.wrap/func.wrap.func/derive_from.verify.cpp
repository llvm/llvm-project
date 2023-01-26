//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <functional>
//
// Make sure we can't initialize a std::function using an allocator (http://wg21.link/p0302r1).
// These constructors were removed in C++17.

#include <functional>
#include <memory>

struct S : public std::function<void()> { using function::function; };

void f() {
  S f1( [](){} );
  S f2(std::allocator_arg, std::allocator<int>{}, f1); // expected-error {{no matching constructor for initialization of 'S'}}
}
