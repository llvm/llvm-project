//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

#include <variant>

#include "test_macros.h"

using A1 [[maybe_unused]] = std::hash<std::variant<int, long>>::argument_type;
using R1 [[maybe_unused]] = std::hash<std::variant<int, long>>::result_type;
#if TEST_STD_VER >= 20
// expected-error@-3 {{no type named 'argument_type' in 'std::hash<std::variant<int, long>>'}}
// expected-error@-3 {{no type named 'result_type' in 'std::hash<std::variant<int, long>>'}}
#else
// expected-warning@-6 {{'argument_type' is deprecated}}
// expected-warning@-6 {{'result_type' is deprecated}}
#endif

using A2 [[maybe_unused]] = std::hash<std::monostate>::argument_type;
using R2 [[maybe_unused]] = std::hash<std::monostate>::result_type;
#if TEST_STD_VER >= 20
// expected-error@-3 {{no type named 'argument_type' in 'std::hash<monostate>'}}
// expected-error@-3 {{no type named 'result_type' in 'std::hash<monostate>'}}
#else
// expected-warning@-6 {{'argument_type' is deprecated}}
// expected-warning@-6 {{'result_type' is deprecated}}
#endif
