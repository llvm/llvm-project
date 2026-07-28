//===-- unordsf2.S - single-precision floating point comparison -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Return 1 if the result of comparing x with y is 'unordered', i.e.
// one of x and y is NaN.
//
//===----------------------------------------------------------------------===//

#include "../../assembly.h"

  .syntax unified
  .text
  .p2align 2

DEFINE_COMPILERRT_FUNCTION_ALIAS(__unordsf2, __aeabi_fcmpun)

DEFINE_COMPILERRT_THUMB_FUNCTION(__aeabi_fcmpun)

  // This function isn't based on the general-purpose code in fcmp.h, because
  // it's more effort than needed. Here we just need to identify whether or not
  // there's at least one NaN in the inputs. There's no need to vary that check
  // based on the sign bit, so we might as well just do the NaN test as quickly
  // as possible.
  movs    r2, #0xFF
  lsls    r2, r2, #24
  lsls    r3, r0, #1
  cmp     r3, r2
  bhi     LOCAL_LABEL(NaN)
  lsls    r3, r1, #1
  cmp     r3, r2
  bhi     LOCAL_LABEL(NaN)

  // If HS, then we have no NaNs and return false.
  movs    r0, #0
  bx      lr

  // Otherwise, we have at least one NaN, and return true.
LOCAL_LABEL(NaN):
  movs    r0, #1
  bx      lr

END_COMPILERRT_FUNCTION(__aeabi_fcmpun)

NO_EXEC_STACK_DIRECTIVE
