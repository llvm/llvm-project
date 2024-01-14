//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_CHECK_CONSTEXPR_H
#define TEST_SUPPORT_CHECK_CONSTEXPR_H

#include "test_macros.h"

#if TEST_STD_VER < 11
#  error "C++11 or greater is required to use this header"
#endif

#define TEST_EXPRESSION_CONSTEXPR(expr) static_assert(__builtin_constant_p(expr))
#define TEST_EXPRESSION_NOT_CONSTEXPR(expr) static_assert(!__builtin_constant_p(expr));

#endif // TEST_SUPPORT_CHECK_CONSTEXPR_H
