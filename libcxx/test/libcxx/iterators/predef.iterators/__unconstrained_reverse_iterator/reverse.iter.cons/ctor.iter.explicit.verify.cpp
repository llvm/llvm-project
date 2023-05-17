//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// __unconstrained_reverse_iterator

// explicit __unconstrained_reverse_iterator(Iter x);

// test explicitness

#include <iterator>

void f() {
    char const* it = "";
    std::__unconstrained_reverse_iterator<char const*> r = it; // expected-error{{no viable conversion from 'const char *' to 'std::__unconstrained_reverse_iterator<const char *>'}}
}
