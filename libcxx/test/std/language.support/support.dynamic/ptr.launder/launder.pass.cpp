//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <new>

// template <class T> constexpr T* launder(T* p) noexcept;

// UNSUPPORTED: c++03, c++11, c++14

#include <new>
#include <cassert>

#include "test_macros.h"

constexpr int gi = 5;
constexpr float gf = 8.f;

int main(int, char**) {
    static_assert(std::launder(&gi) == &gi, "" );
    static_assert(std::launder(&gf) == &gf, "" );

    const int *i = &gi;
    const float *f = &gf;
    static_assert(std::is_same<decltype(i), decltype(std::launder(i))>::value, "");
    static_assert(std::is_same<decltype(f), decltype(std::launder(f))>::value, "");

    assert(std::launder(i) == i);
    assert(std::launder(f) == f);

  return 0;
}
