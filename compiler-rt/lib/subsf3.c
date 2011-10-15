//===-- lib/subsf3.c - Single-precision subtraction ---------------*- C -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements single-precision soft-float subtraction with the
// IEEE-754 default rounding (to nearest, ties to even).
//
//===----------------------------------------------------------------------===//
#include "abi.h"

#define SINGLE_PRECISION
#include "fp_lib.h"

fp_t COMPILER_RT_ABI __addsf3(fp_t a, fp_t b);

ARM_EABI_FNALIAS(fsub, subsf3);

// Subtraction; flip the sign bit of b and add.
COMPILER_RT_ABI fp_t
__subsf3(fp_t a, fp_t b) {
    return __addsf3(a, fromRep(toRep(b) ^ signBit));
}

/* FIXME: rsub for ARM EABI */
