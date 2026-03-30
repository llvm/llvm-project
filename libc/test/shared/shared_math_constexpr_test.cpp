//===-- Unittests for shared math functions in constexpr context ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define LIBC_ENABLE_CONSTEXPR 1

#include "shared/math.h"

//===----------------------------------------------------------------------===//
//                       Double Tests
//===----------------------------------------------------------------------===//

static_assert(0.0 == LIBC_NAMESPACE::shared::ceil(0.0));
static_assert(0.0 == LIBC_NAMESPACE::shared::log(1.0));

//===----------------------------------------------------------------------===//
//                       Float Tests
//===----------------------------------------------------------------------===//

static_assert(0.0f == LIBC_NAMESPACE::shared::ceilf(0.0f));

//===----------------------------------------------------------------------===//
//                       Float16 Tests
//===----------------------------------------------------------------------===//

#ifdef LIBC_TYPES_HAS_FLOAT16

static_assert(0.0f16 == LIBC_NAMESPACE::shared::ceilf16(0.0f16));

#endif // LIBC_TYPES_HAS_FLOAT16

//===----------------------------------------------------------------------===//
//                       Long Double Tests
//===----------------------------------------------------------------------===//

// TODO(issue#185232): Mark as constexpr once the refactor is done.
#if 0 // Temporarily disable long double tests

static_assert(0.0L == LIBC_NAMESPACE::shared::ceill(0.0L));

#endif

//===----------------------------------------------------------------------===//
//                       Float128 Tests
//===----------------------------------------------------------------------===//

#ifdef LIBC_TYPES_HAS_FLOAT128

static_assert(float128(0.0) == LIBC_NAMESPACE::shared::ceilf128(float128(0.0)));

#endif // LIBC_TYPES_HAS_FLOAT128

//===----------------------------------------------------------------------===//
//                       BFloat16 Tests
//===----------------------------------------------------------------------===//

static_assert(bfloat16(0.0) == LIBC_NAMESPACE::shared::ceilbf16(bfloat16(0.0)));
