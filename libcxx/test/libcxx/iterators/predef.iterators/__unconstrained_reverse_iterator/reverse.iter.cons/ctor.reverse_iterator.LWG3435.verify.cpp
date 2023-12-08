//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// __unconstrained_reverse_iterator

// template <class U>
//  requires !same_as<U, Iter> && convertible_to<const U&, Iter>
// __unconstrained_reverse_iterator(const __unconstrained_reverse_iterator<U> &);

#include <iterator>

struct Base { };
struct Derived : Base { };

void test() {
    std::__unconstrained_reverse_iterator<Base*> base;
    std::__unconstrained_reverse_iterator<Derived*> derived(base); // expected-error {{no matching constructor for initialization of 'std::__unconstrained_reverse_iterator<Derived *>'}}
}
