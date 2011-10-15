//===-- lib/adddf3.c - Double-precision subtraction ---------------*- C -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements double-precision soft-float subtraction with the
// IEEE-754 default rounding (to nearest, ties to even).
//
//===----------------------------------------------------------------------===//
#include "abi.h"

#define DOUBLE_PRECISION
#include "fp_lib.h"

fp_t COMPILER_RT_ABI __adddf3(fp_t a, fp_t b);


ARM_EABI_FNALIAS(dsub, subdf3);

// Subtraction; flip the sign bit of b and add.
COMPILER_RT_ABI fp_t
__subdf3(fp_t a, fp_t b) {
    return __adddf3(a, fromRep(toRep(b) ^ signBit));
}

/* FIXME: rsub for ARM EABI */
