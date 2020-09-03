//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <functional>
//
// reference_wrapper
//
// template <class... ArgTypes>
//  std::invoke_result_t<T&, ArgTypes...>
//      operator()(ArgTypes&&... args) const;
//
// Requires T to be a complete type (since C++20).

#include <functional>


struct Foo;
Foo& get_foo();

void test() {
    std::reference_wrapper<Foo> ref = get_foo();
    ref(0); // incomplete at the point of call
}

struct Foo { void operator()(int) const { } };
Foo& get_foo() { static Foo foo; return foo; }

int main() {
    test();
}
