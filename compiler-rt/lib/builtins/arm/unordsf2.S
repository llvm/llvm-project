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

#include "../assembly.h"


  .syntax unified
  .text
  .p2align 2

#if __ARM_PCS_VFP
DEFINE_COMPILERRT_FUNCTION(__unordsf2)
  push {r4, lr}
  vmov r0, s0
  vmov r1, s1
  bl __aeabi_fcmpun
  pop {r4, pc}
#else
DEFINE_COMPILERRT_FUNCTION_ALIAS(__unordsf2, __aeabi_fcmpun)
#endif

DEFINE_COMPILERRT_FUNCTION(__aeabi_fcmpun)

  // This function isn't based on the general-purpose code in fcmp.h, because
  // it's more effort than needed. Here we just need to identify whether or not
  // there's at least one NaN in the inputs. There's no need to vary that check
  // based on the sign bit, so we might as well just do the NaN test as quickly
  // as possible.
  mov     r12, #0xFF << 24
  cmp     r12, r0, lsl #1    // if LO, then r12 < (r0 << 1), so r0 is a NaN
  cmphs   r12, r1, lsl #1    // if not LO, then do the same check for r1

  // If HS, then we have no NaNs and return false. We do this as quickly as we
  // can (not stopping to take two instructions setting up r0 for both
  // possibilities), on the assumption that NaNs are rare and we want to
  // optimize for the non-NaN path.
  movhs   r0, #0
  bxhs    lr

  // Otherwise, we have at least one NaN, and return true.
  mov     r0, #1
  bx      lr

END_COMPILERRT_FUNCTION(__aeabi_fcmpun)

NO_EXEC_STACK_DIRECTIVE
