//===-- gedf2.S - double-precision floating point comparison --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This function has the semantics of GNU __cmpdf2, except for its NaN
// handling. It's a three-way compare which returns <0 if x<y, 0 if x==y, and
// >0 if x>y. If the result is unordered (i.e. x or y or both is NaN) then it
// returns <0, where __cmpdf2 would return >0.
//
// This also makes it suitable for use as __gtdf2 or __gedf2 (or __eqdf2 or
// __nedf2).
//
//===----------------------------------------------------------------------===//

#include "../../assembly.h"
#include "../endian.h"

  .syntax unified
  .text
  .p2align 2

op0h .req xh
op0l .req xl
op1h .req yh
op1l .req yl
.macro SetReturnRegister
  bhi 0f
  blo 1f
  movs r0, #0
  // This macro is always called immediately before returning from the
  // function, so it's safe to use the same return instruction here, instead of
  // wasting time branching forward to the end of the macro.
  pop     {r4,r5,r6,pc}
0:
  movs r0, #1
  pop     {r4,r5,r6,pc}
1:
  movs r0, #1
  rsbs r0, r0, #0
  pop     {r4,r5,r6,pc}
.endm

DEFINE_COMPILERRT_FUNCTION_ALIAS(__gedf2, __compiler_rt_softfp_gedf2)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__gtdf2, __gedf2)

DEFINE_COMPILERRT_THUMB_FUNCTION(__compiler_rt_softfp_gedf2)
  #include "dcmp.h"

LOCAL_LABEL(NaN):
  movs r0, #1
  rsbs r0, r0, #0
  pop     {r4,r5,r6,pc}

END_COMPILERRT_FUNCTION(__compiler_rt_softfp_gedf2)

NO_EXEC_STACK_DIRECTIVE
