//===-- floatunssisf.S - 32-bit unsigned int to single-precision FP conversion//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the __floatunssisf function (32-bit unsigned integer to
// single precision floating point conversion), with the IEEE-754 default
// rounding (to nearest, ties to even), for the Arm and Thumb2 ISAs.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

  .syntax unified
  .text
  .p2align 2

#if __ARM_PCS_VFP
DEFINE_COMPILERRT_FUNCTION(__floatunssisf)
  push {r4, lr}
  bl __aeabi_ui2f
  vmov s0, r0
  pop {r4, pc}
#else
DEFINE_COMPILERRT_FUNCTION_ALIAS(__floatunssisf, __aeabi_ui2f)
#endif

DEFINE_COMPILERRT_FUNCTION(__aeabi_ui2f)

  // Find the highest set bit of the input, and shift it up to the top bit. r3
  // contains the amount we shifted by, and r2 the shifted value.
  clz     r3, r0
  lsls    r2, r0, r3

  // Convert the shift distance into the exponent of the output float. The
  // exponent for an integer with bit 31 set should be 0x7f (the IEEE exponent
  // bias) plus 31, which is 0x9e. Here we reduce that by 1, because when we
  // add the mantissa to it, the leading mantissa bit will increment it.
  rsb     r3, r3, #0x9d

  // A side effect of the LSLS above was to set the Z flag if the input integer
  // was actually zero. In that situation, we can just return immediately,
  // because r0 still _contains_ the input integer, which has the same
  // representation as the floating-point +0 that we should return.
#if !__thumb__
  bxeq    lr
#else
  // In Thumb, we do the conditional return by branching to a return
  // instruction later in the function. This costs more time in the case where
  // the return is taken, but saves an IT in the case where it's not, and we
  // assume that nonzero integers are converted to FP more often than zero is.
  // (This also improves the _worst-case_ running time, because the nonzero
  // code path is the limiting factor.)
  beq     LOCAL_LABEL(return)
#endif

  // Shift the exponent up to its final bit position.
  lsls    r1, r3, #23

  // Recombine the mantissa with the exponent, and round. This is done
  // differently between Arm and Thumb.
#if !__thumb__
  // Arm rounding sequence: shift the round bit off the top of r2 into C, and
  // simultaneously set Z if the lower-down bits are all zero.
  lsls    r3, r2, #25
  // Recombine mantissa with exponent, using ADC so that this also adds 1 if
  // we're rounding up.
  adc     r0, r1, r2, lsr #8
  // If C was clear, we can't possibly need to round to even, so return.
  bxcc    lr
  // Now we've definitely rounded up, and if Z is set, round to even.
  biceq   r0, r0, #1
  bx      lr
#else
  // Thumb rounding sequence: we do things in a slightly different order, by
  // recombining first with plain ADD, and _then_ testing the round bit. On
  // simple M-profile CPUs like Cortex-M3, this avoids the IT instruction
  // (inserted before BXCC lr) costing a cycle, because it immediately follows
  // a 16-bit LSLS instruction, so the CPU had already fetched it.
  //
  // So we save a cycle in the case where we don't round up, at the cost of a
  // cycle in the case where we do (requiring a separate ADD instruction after
  // the BXCC lr isn't taken). We expect that this is a good trade, on the
  // theory that _most_ integers converted into floating point are not large
  // enough to need rounding at all, so all the exact cases _and_ half the
  // inexact ones will benefit from the saving.
  add     r0, r1, r2, lsr #8    // r0 is now exp+mant, unrounded
  lsls.N  r3, r2, #25           // .N to make sure it's assembled as 16-bit
  bxcc    lr
  // Now if we didn't take the return, we must definitely round up, and
  // conditionally round to even.
  add     r0, r0, #1
  biceq   r0, r0, #1
LOCAL_LABEL(return):                    // label we branch to from the 0 case above
  bx      lr
#endif

END_COMPILERRT_FUNCTION(__aeabi_ui2f)

NO_EXEC_STACK_DIRECTIVE
