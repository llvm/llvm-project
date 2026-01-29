//===-- cmpsf2.S - single-precision floating point comparison -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This function has the semantics of GNU __cmpsf2: it's a three-way compare
// which returns <0 if x<y, 0 if x==y, and >0 if x>y. If the result is
// unordered (i.e. x or y or both is NaN) then it returns >0.
//
// This also makes it suitable for use as all of __eqsf2, __nesf2, __ltsf2 or
// __lesf2.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

  .syntax unified
  .text
  .p2align 2

op0 .req r0
op1 .req r1
.macro SetReturnRegister
  mov r0, #0
  movhi r0, #1
  movlo r0, #-1
.endm

#if __ARM_PCS_VFP
DEFINE_COMPILERRT_FUNCTION(__cmpsf2)
  push {r4, lr}
  vmov r0, s0
  vmov r1, s1
  bl __compiler_rt_softfp_cmpsf2
  pop {r4, pc}
#else
DEFINE_COMPILERRT_FUNCTION_ALIAS(__cmpsf2, __compiler_rt_softfp_cmpsf2)
#endif
DEFINE_COMPILERRT_FUNCTION_ALIAS(__lesf2, __cmpsf2)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__ltsf2, __cmpsf2)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__eqsf2, __cmpsf2)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__nesf2, __cmpsf2)

DEFINE_COMPILERRT_FUNCTION(__compiler_rt_softfp_cmpsf2)
  #include "fcmp.h"

LOCAL_LABEL(NaN):
  mov r0, #+1
  bx lr

END_COMPILERRT_FUNCTION(__compiler_rt_softfp_cmpsf2)

NO_EXEC_STACK_DIRECTIVE
