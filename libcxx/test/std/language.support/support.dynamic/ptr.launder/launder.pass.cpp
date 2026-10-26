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

#include <cassert>
#include <new>
#include <type_traits>

constexpr int gi = 5;
constexpr float gf = 8.f;

struct A { int x, y; };
A a[100];

// LWG2859. Visible case where the optimiser sees everything and must not drop the store
// to a[3].
int test_reach_visible() {
    a[3].y = 1;
    int* p = std::launder(&a[2].x);
    A*   q = reinterpret_cast<A*>(p) + 1;
    q->y = 42;
    return a[3].y;
}

int main(int, char**) {
    static_assert(std::launder(&gi) == &gi, "" );
    static_assert(std::launder(&gf) == &gf, "" );

    const int *i = &gi;
    const float *f = &gf;
    static_assert(std::is_same<decltype(i), decltype(std::launder(i))>::value, "");
    static_assert(std::is_same<decltype(f), decltype(std::launder(f))>::value, "");

    assert(std::launder(i) == i);
    assert(std::launder(f) == f);

    assert(test_reach_visible() == 42);

  return 0;
}
