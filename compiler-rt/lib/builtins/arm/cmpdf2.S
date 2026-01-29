//===-- cmpdf2.S - double-precision floating point comparison -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This function has the semantics of GNU __cmpdf2: it's a three-way compare
// which returns <0 if x<y, 0 if x==y, and >0 if x>y. If the result is
// unordered (i.e. x or y or both is NaN) then it returns >0.
//
// This also makes it suitable for use as all of __eqdf2, __nedf2, __ltdf2 or
// __ledf2.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"
#include "endian.h"

  .syntax unified
  .text
  .p2align 2


op0h .req xh
op0l .req xl
op1h .req yh
op1l .req yl
.macro SetReturnRegister
  mov r0, #0
  movhi r0, #1
  movlo r0, #-1
.endm
.macro SetReturnRegisterNE
  movne r0, #-1
  movhi r0, #1
.endm

#if __ARM_PCS_VFP
DEFINE_COMPILERRT_FUNCTION(__cmpdf2)
  push {r4, lr}
  VMOV_FROM_DOUBLE(r0, r1, d0)
  VMOV_FROM_DOUBLE(r2, r3, d1)
  bl __compiler_rt_softfp_cmpdf2
  pop {r4, pc}
#else
DEFINE_COMPILERRT_FUNCTION_ALIAS(__cmpdf2, __compiler_rt_softfp_cmpdf2)
#endif
DEFINE_COMPILERRT_FUNCTION_ALIAS(__ledf2, __cmpdf2)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__ltdf2, __cmpdf2)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__eqdf2, __cmpdf2)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__nedf2, __cmpdf2)

DEFINE_COMPILERRT_FUNCTION(__compiler_rt_softfp_cmpdf2)
  #include "dcmp.h"

LOCAL_LABEL(NaN):
  mov r0, #+1
  bx lr

END_COMPILERRT_FUNCTION(__compiler_rt_softfp_cmpdf2)

NO_EXEC_STACK_DIRECTIVE
