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

// expected-error@27{{'N::midpoint' has different definitions in different modules; defined here first difference is 1st parameter with type 'F'}}
// expected-error@24{{'N::midpoint' has different definitions in different modules; defined here first difference is 1st parameter with type 'U'}}
// expected-note@21{{but in '' found 1st parameter with type 'T'}}
int x = N::something;
// expected-error@37{{no member named 'something' in namespace 'N'}}
// expected-note@21{{but in '' found 1st parameter with type 'T'}}

#endif
