// RUN: %clang_cc1 -fallow-pch-with-compiler-errors -std=c++20 -x c++-header -emit-pch %s -o %t -verify
// RUN: %clang_cc1 -fallow-pch-with-compiler-errors -std=c++20 -include-pch %t %s -verify
#ifndef HEADER_H
#define HEADER_H

#include "bad_include.h"
// expected-error@6{{'bad_include.h' file not found}}

template <bool, class = void> struct enable_if {};
template <class T> struct enable_if<true, T> { typedef T type; };
template <bool B, class T = void> using enable_if_t = typename enable_if<B, T>::type;

template <typename> struct meta { static constexpr int value = 0; };
template <> struct meta<int> { static constexpr int value = 1; };
template <> struct meta<float> { static constexpr int value = 2; };

namespace N {
inline namespace inner {

template <class T>
constexpr enable_if_t<meta<T>::value == 0, void> midpoint(T) {}

template <class U>
constexpr enable_if_t<meta<U>::value == 1, void> midpoint(U) {}

template <class F>
constexpr enable_if_t<meta<F>::value == 2, void> midpoint(F) {}

} // namespace inner
} // namespace N

#else

// FIXME: Change the test to trigger a suitable error: previously the test
// asserted the ODR error ("'N::midpoint' has different definitions in different
// modules"), which isn't fully correct as there's only one module, and since a
// change in the ODR hash calculation this error isn't triggered anymore.

int x = N::something;
// expected-error@39{{no member named 'something' in namespace 'N'}}

#endif
