//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// These compilers don't support std::reference_converts_from_temporary yet.
// UNSUPPORTED: android, apple-clang-15, apple-clang-16, clang-18, clang-19.1

// <type_traits>

// template<class T, class U> struct reference_converts_from_temporary;

// template<class T, class U>
// constexpr bool reference_converts_from_temporary_v
//   = reference_converts_from_temporary<T, U>::value;

#include <type_traits>

#if !__has_warning("-Winvalid-specialization")
// expected-no-diagnostics
#else

// expected-error@+2 {{'reference_converts_from_temporary' cannot be specialized: Users are not allowed to specialize this standard library entity}}
template <>
struct std::reference_converts_from_temporary<int, int> {
}; //expected-error 0-1 {{explicit specialization of 'std::reference_converts_from_temporary<int, int>' after instantiation}}

// expected-error@+2 {{'reference_converts_from_temporary' cannot be specialized: Users are not allowed to specialize this standard library entity}}
template <typename T, typename U>
struct std::reference_converts_from_temporary<T&, U&> {};

// expected-error@+2 {{'reference_converts_from_temporary_v' cannot be specialized: Users are not allowed to specialize this standard library entity}}
template <>
inline constexpr bool std::reference_converts_from_temporary_v<int, int> =
    false; // expected-error 0-1 {{explicit specialization of 'std::reference_converts_from_temporary_v<int, int>' after instantiation}}

// expected-error@+2 {{'reference_converts_from_temporary_v' cannot be specialized: Users are not allowed to specialize this standard library entity}}
template <class T, class U>
inline constexpr bool std::reference_converts_from_temporary_v<T&, U&> = false;

#endif
