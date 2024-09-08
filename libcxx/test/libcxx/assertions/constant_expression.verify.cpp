//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that both `_LIBCPP_ASSERT(false, ...)` and `_LIBCPP_ASSUME(false)`
// mean that a constant expression cannot be formed.

#include <__assert>
#include "test_macros.h"

// expected-note@*:* 0+ {{expanded from macro}}

static_assert((_LIBCPP_ASSERT(false, "message"), true), "");
// expected-error@-1 {{static assertion expression is not an integral constant expression}}

static_assert((_LIBCPP_ASSUME(false), true), "");
// expected-error@-1 {{static assertion expression is not an integral constant expression}}

static_assert(!__builtin_constant_p(_LIBCPP_ASSERT(false, "message")), "");
static_assert(!__builtin_constant_p(_LIBCPP_ASSUME(false)), "");
