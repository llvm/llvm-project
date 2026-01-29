//===-- endian.h - make double-prec software FP work in both endiannesses -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file should be included from assembly source code (not C). It
// defines two pairs of register aliases, for handling 64-bit values passed and
// returned from functions in the AArch32 integer registers:
//
//   xh, xl      the high and low words of a 64-bit value passed in {r0,r1}
//   yh, yl      the high and low words of a 64-bit value passed in {r2,r3}
//
// Which alias goes with which register depends on endianness.
//
//===----------------------------------------------------------------------===//

#ifndef COMPILER_RT_ARM_FP_ENDIAN_H
#define COMPILER_RT_ARM_FP_ENDIAN_H

#ifdef __BIG_ENDIAN__
// Big-endian: high words are in lower-numbered registers.
xh .req r0
xl .req r1
yh .req r2
yl .req r3
#else
// Little-endian: low words are in lower-numbered registers.
xl .req r0
xh .req r1
yl .req r2
yh .req r3
#endif

#endif // COMPILER_RT_ARM_FP_ENDIAN_H
