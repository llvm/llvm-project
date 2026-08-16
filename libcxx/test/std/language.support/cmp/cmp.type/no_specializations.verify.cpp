//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// These compilers do not support __builtin_type_order
// UNSUPPORTED: clang-21, clang-22, clang-23, apple-clang-21
// UNSUPPORTED: gcc-15

// <compare>

// template<class T, class U>
//   struct type_order;

#include <compare>

#if !__has_warning("-Winvalid-specializations")
// expected-no-diagnostics
#else
template <>
struct std::type_order<int, int>; // expected-error {{cannot be specialized}}

template <class T>
struct std::type_order<T, int>; // expected-error {{cannot be specialized}}
#endif
