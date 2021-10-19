//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// explicit move_iterator(Iter );

// test explicitness

#include <iterator>

int main(int, char**) {
    char const* it = "";
    std::move_iterator<char const*> r = it; // expected-error{{no viable conversion from 'const char *' to 'std::move_iterator<const char *>'}}
    return 0;
}
