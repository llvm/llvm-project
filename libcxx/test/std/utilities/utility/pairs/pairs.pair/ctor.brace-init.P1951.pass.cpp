//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: c++11 || c++14 || c++17 || c++20

// This test makes sure that we don't apply P1951 before C++23, since that is
// a breaking change. The examples in this test are taken from Richard Smith's
// comments on https://llvm.org/D109066.

#include <cassert>
#include <utility>
#include <vector>

struct A {
    int *p_;
    A(int *p) : p_(p) { *p_ += 1; }
    A(const A& a) : p_(a.p_) { *p_ += 1; }
    ~A() { *p_ -= 1; }
};

int main(int, char**) {
    // Example 1:
    // Without P1951, we call the `pair(int, const A&)` constructor (the converting constructor is not usable because
    // we can't deduce from an initializer list), which creates the A temporary as part of the call to f. With P1951,
    // we call the `pair(U&&, V&&)` constructor, which creates a A temporary inside the pair constructor, and that
    // temporary doesn't live long enough any more.
    {
        int i = 0;
        auto f = [&](std::pair<std::vector<int>, const A&>) { assert(i >= 1); };
        f({{42, 43}, &i});
    }

    // Example 2:
    // Here, n doesn't need to be captured if we call the `pair(const int&, const long&)` constructor, because
    // the lvalue-to-rvalue conversion happens in the lambda. But if we call the `pair(U&&, V&&)` constructor
    // (deducing V = int), then n does need to be captured.
    {
        const int n = 5;
        (void) []{ std::pair<int, long>({1}, n); };
    }

    return 0;
}
