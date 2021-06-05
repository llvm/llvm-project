//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// explicit reverse_iterator(Iter x);

// test explicitness

#include <iterator>

int main(int, char**) {
    char const* it = "";
    std::reverse_iterator<char const*> r = it; // expected-error{{no viable conversion from 'const char *' to 'std::reverse_iterator<const char *>'}}
    return 0;
}
