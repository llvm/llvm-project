//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Check that user-specializations are diagnosed
// See [cmp.result]/1

#include <compare>

#include "test_macros.h"

#if !__has_warning("-Winvalid-specialization")
// expected-no-diagnostics
#else
struct S {};

template <>
struct std::compare_three_way_result<S>; // expected-error {{cannot be specialized}}

#  if TEST_STD_VER > 23 && __has_builtin(__builtin_type_order)
template <>
struct std::type_order<int, int>; // expected-error {{cannot be specialized}}

template <class T>
struct std::type_order<T, int>; // expected-error {{cannot be specialized}}

template <>
constexpr std::strong_ordering std::type_order_v<int, int> = false; // expected-error {{cannot be specialized}}

template <class T>
constexpr std::strong_ordering std::type_order_v<T, int> = false; // expected-error {{cannot be specialized}}
#  endif
#endif
