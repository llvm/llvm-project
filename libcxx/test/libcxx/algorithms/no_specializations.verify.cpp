//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// Check that user-specializations are diagnosed
// See [execpol.type]/3

#include <execution>

#if !__has_warning("-Winvalid-specializations")
// expected-no-diagnostics
#else
struct S {};

template <>
struct std::is_execution_policy<S>; // expected-error {{cannot be specialized}}

template <>
constexpr bool std::is_execution_policy_v<S> = false; // expected-error {{cannot be specialized}}
#endif
