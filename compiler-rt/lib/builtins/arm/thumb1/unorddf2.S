//===-- unorddf2.S - double-precision floating point comparison -----------===//
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
#include "../endian.h"

  .syntax unified
  .text
  .p2align 2

DEFINE_COMPILERRT_FUNCTION_ALIAS(__unorddf2, __aeabi_dcmpun)

DEFINE_COMPILERRT_THUMB_FUNCTION(__aeabi_dcmpun)

  // This function isn't based on the general-purpose code in dcmp.h, because
  // it's more effort than needed. Here we just need to identify whether or not
  // there's at least one NaN in the inputs. There's no need to vary that check
  // based on the sign bit, so we might as well just do the NaN test as quickly
  // as possible.
  //
  // We do this by shifting the high word of each input left to get rid of the
  // sign bit, shifting a bit in at the bottom which is 1 if any bit is set in
  // the low word. Then we check if the result is _greater_ than 0xFFE00000
  // (but not equal), via adding 0x00200000 to it and testing for the HI
  // condition (carry flag set, but Z clear).
  //
  // Once we've done that transformation to the first input xh:xl, we
  // free up xl to contain our constant 0x00200000, so there's no need
  // to push any registers.
  cmp     xl, #1                // set C if xl is nonzero
  adcs    xh, xh, xh            // shift xh left, bringing in the C bit
  movs    xl, #1                // now xl is free, make the test constant
  lsls    xl, xl, #21           //   by shifting 1 left to make 0x00200000
  cmn     xh, xl                // HI if x is a NaN
  bhi     LOCAL_LABEL(NaN)
  cmp     yl, #1                // set C if yl is nonzero
  adcs    yh, yh, yh            // shift yh left, bringing in the C bit
  cmn     yh, xl                // HI if y is a NaN
  bhi     LOCAL_LABEL(NaN)

  movs    r0, #0
  bx      lr

LOCAL_LABEL(NaN):
  movs    r0, #1
  bx      lr

END_COMPILERRT_FUNCTION(__aeabi_dcmpun)

NO_EXEC_STACK_DIRECTIVE
