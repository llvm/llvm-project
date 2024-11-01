//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <functional>

// class function<R(ArgTypes...)>

// R operator()(ArgTypes... args) const

#include <functional>
#include <cassert>

// member data pointer: cv qualifiers should transfer from argument to return type

struct Foo { int data; };

void f() {
    int Foo::*fp = &Foo::data;
    std::function<int& (const Foo*)> r2(fp); // expected-error {{no matching constructor for initialization of}}
}
