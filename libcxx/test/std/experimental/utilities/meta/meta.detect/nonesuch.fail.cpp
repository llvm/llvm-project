//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <experimental/type_traits>
//
//  struct nonesuch;
//	  nonesuch has no default constructor (C++17 §15.1)
//    or initializer-list constructor (C++17 §11.6.4),
//    and is not an aggregate (C++17 §11.6.1).


#include <experimental/type_traits>
#include <string>

#include "test_macros.h"

namespace ex = std::experimental;

void doSomething (const ex::nonesuch &) {}

int main(int, char**) {
    ex::nonesuch *e0 = new ex::nonesuch; // expected-error {{no matching constructor for initialization of 'ex::nonesuch'}}
    doSomething({}); // expected-error{{no matching function for call to 'doSomething'}}

    return 0;
}
