//===-- Unittests for shared math functions in constexpr context ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define LIBC_ENABLE_CONSTEXPR 1

#include "shared/math.h"
#include "test/UnitTest/Test.h"

//===----------------------------------------------------------------------===//
//                       Double Tests
//===----------------------------------------------------------------------===//

static_assert(0.0 == LIBC_NAMESPACE::shared::ceil(0.0));
static_assert(0.0 == LIBC_NAMESPACE::shared::copysign(0.0, 0.0));
static_assert(1.0 == LIBC_NAMESPACE::shared::fabs(-1.0));
static_assert(1.0 == LIBC_NAMESPACE::shared::floor(1.2));
static_assert(0.0 == LIBC_NAMESPACE::shared::log(1.0));

//===----------------------------------------------------------------------===//
//                       Float Tests
//===----------------------------------------------------------------------===//

static_assert(0.0f == LIBC_NAMESPACE::shared::ceilf(0.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::copysignf(0.0f, 0.0f));
static_assert(1.0f == LIBC_NAMESPACE::shared::fabsf(-1.0f));
static_assert(0.0f == LIBC_NAMESPACE::shared::floorf(0.0f));

//===----------------------------------------------------------------------===//
//                       Float16 Tests
//===----------------------------------------------------------------------===//

#ifdef LIBC_TYPES_HAS_FLOAT16

static_assert(0.0f16 == LIBC_NAMESPACE::shared::ceilf16(0.0f16));
static_assert(0.0f16 == LIBC_NAMESPACE::shared::copysignf16(0.0f16, 0.0f16));
static_assert(1.0f16 == LIBC_NAMESPACE::shared::fabsf16(-1.0f16));
static_assert(3.0f16 == LIBC_NAMESPACE::shared::floorf16(3.7f16));

#endif // LIBC_TYPES_HAS_FLOAT16

//===----------------------------------------------------------------------===//
//                       Long Double Tests
//===----------------------------------------------------------------------===//

// TODO(issue#185232): Mark as constexpr once the refactor is done.
#if 0 // Temporarily disable long double tests

static_assert(0.0L == LIBC_NAMESPACE::shared::ceill(0.0L));
static_assert(0.0L == LIBC_NAMESPACE::shared::copysignl(0.0L, 0.0L));
static_assert(0.0 == LIBC_NAMESPACE::shared::ddivl(0.0L, 1.0L));
static_assert(0.0 == LIBC_NAMESPACE::shared::dmull(0.0L, 1.0L));
static_assert(1.0L == LIBC_NAMESPACE::shared::fabsl(-1.0L));
static_assert(0.0L == LIBC_NAMESPACE::shared::floorl(0.0L));

#endif

//===----------------------------------------------------------------------===//
//                       Float128 Tests
//===----------------------------------------------------------------------===//

#ifdef LIBC_TYPES_HAS_FLOAT128

static_assert(float128(0.0) == LIBC_NAMESPACE::shared::ceilf128(float128(0.0)));
static_assert(float128(1.0) ==
              LIBC_NAMESPACE::shared::fabsf128(float128(-1.0)));
static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::copysignf128(float128(0.0),
                                                   float128(0.0)));
static_assert(0.0 ==
              LIBC_NAMESPACE::shared::ddivf128(float128(0.0), float128(1.0)));
static_assert(0.0 ==
              LIBC_NAMESPACE::shared::dmulf128(float128(0.0), float128(1.0)));
static_assert(float128(0.0) ==
              LIBC_NAMESPACE::shared::floorf128(float128(0.0)));

#endif // LIBC_TYPES_HAS_FLOAT128

//===----------------------------------------------------------------------===//
//                       BFloat16 Tests
//===----------------------------------------------------------------------===//

static_assert(bfloat16(0.0) == LIBC_NAMESPACE::shared::asinbf16(bfloat16(0.0)));
static_assert(bfloat16(0.0) == LIBC_NAMESPACE::shared::ceilbf16(bfloat16(0.0)));
static_assert(bfloat16(1.0) ==
              LIBC_NAMESPACE::shared::fabsbf16(bfloat16(-1.0)));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::copysignbf16(bfloat16(0.0),
                                                   bfloat16(0.0)));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::floorbf16(bfloat16(0.0f)));
static_assert(bfloat16(0.0) ==
              LIBC_NAMESPACE::shared::logbbf16(bfloat16(1.0f)));

TEST(LlvmLibcSharedMathTest, ConstantEvaluation) {}
