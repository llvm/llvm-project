//===-- int_mul_impl.inc - Integer multiplication -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers used by __mulsi3, __muldi3.
//
//===----------------------------------------------------------------------===//

#include "assembly.h"

#ifndef __mulxi3
#error "__mulxi3 must be defined to use this generic implementation"
#endif

DEFINE_COMPILERRT_FUNCTION(__mulxi3)
  mv     a2, a0
  mv     a0, zero
LOCAL_SYMBOL(loop):
  andi   a3, a1, 1
  beqz   a3, LOCAL_SYMBOL(cond_add)
  add    a0, a0, a2
LOCAL_SYMBOL(cond_add):
  srli   a1, a1, 1
  slli   a2, a2, 1
  bnez   a1, LOCAL_SYMBOL(loop)
  ret
END_COMPILERRT_FUNCTION(__mulxi3)
