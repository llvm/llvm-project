//===-- Unit tests for builtin __bf16 type --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/macros/properties/types.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#if defined(LIBC_TYPES_HAS_BUILTIN_BFLOAT16)

using BFloat16 = LIBC_NAMESPACE::fputil::BFloat16;

TEST(LlvmLibcBuiltinBFloat16Test, TypeSizeAndAlignment) {
  static_assert(sizeof(__bf16) == 2);
  static_assert(alignof(__bf16) == 2);
  __bf16 a = 1.0;
  double b = 1.0;
  static_assert(__bf16(b) == a);
}

#endif // LIBC_TYPES_HAS_BUILTIN_BFLOAT16
