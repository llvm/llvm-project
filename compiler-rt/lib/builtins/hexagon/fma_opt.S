//===----------------------Hexagon builtin routine ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

.macro FUNCTION_BEGIN name
.text
.p2align 5
.globl \name
.type  \name, @function
\name:
.endm

.macro FUNCTION_END name
.size  \name, . - \name
.endm

FUNCTION_BEGIN fmaf
  r2 += sfmpy(r0, r1)
  {
    r0 = r2
    jumpr r31
  }
FUNCTION_END fmaf

  .globl fmal
  .set fmal, fma
