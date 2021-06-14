//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// template <class U>
//  requires !same_as<U, Iter> && convertible_to<const U&, Iter> && assignable_from<Iter&, const U&>
// move_iterator& operator=(const move_iterator<U>& u);

#include <iterator>

struct Base { };
struct Derived : Base { };

void test() {
    std::move_iterator<Base*> base;
    std::move_iterator<Derived*> derived;
    derived = base; // expected-error {{no viable overloaded '='}}
}
