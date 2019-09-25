//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// Before GCC 6, aggregate initialization kicks in.
// See https://stackoverflow.com/q/41799015/627587.
// UNSUPPORTED: gcc-5

// <utility>

// struct piecewise_construct_t { explicit piecewise_construct_t() = default; };
// constexpr piecewise_construct_t piecewise_construct = piecewise_construct_t();

// This test checks for LWG 2510.

#include <utility>


std::piecewise_construct_t f() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}

int main(int, char**) {
    return 0;
}
