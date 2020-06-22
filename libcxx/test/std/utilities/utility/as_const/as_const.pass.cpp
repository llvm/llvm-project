//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template <class T> constexpr add_const<T>& as_const(T& t) noexcept;      // C++17
// template <class T>           add_const<T>& as_const(const T&&) = delete; // C++17

#include <utility>
#include <cassert>

#include "test_macros.h"

struct S {int i;};
bool operator==(const S& x, const S& y) { return x.i == y.i; }
bool operator==(const volatile S& x, const volatile S& y) { return x.i == y.i; }

template<typename T>
void test(T& t)
{
    static_assert(std::is_const<typename std::remove_reference<decltype(std::as_const                  (t))>::type>::value, "");
    static_assert(std::is_const<typename std::remove_reference<decltype(std::as_const<               T>(t))>::type>::value, "");
    static_assert(std::is_const<typename std::remove_reference<decltype(std::as_const<const          T>(t))>::type>::value, "");
    static_assert(std::is_const<typename std::remove_reference<decltype(std::as_const<volatile       T>(t))>::type>::value, "");
    static_assert(std::is_const<typename std::remove_reference<decltype(std::as_const<const volatile T>(t))>::type>::value, "");

    assert(std::as_const(t) == t);
    assert(std::as_const<               T>(t) == t);
    assert(std::as_const<const          T>(t) == t);
    assert(std::as_const<volatile       T>(t) == t);
    assert(std::as_const<const volatile T>(t) == t);
}

int main(int, char**)
{
    int i = 3;
    double d = 4.0;
    S s{2};
    test(i);
    test(d);
    test(s);

  return 0;
}
