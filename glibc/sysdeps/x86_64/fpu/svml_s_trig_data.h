/* Offsets for data table for vectorized sinf, cosf, sincosf.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef S_TRIG_DATA_H
#define S_TRIG_DATA_H

.macro float_vector offset value
/* clang integrated assembler doesn't think subtract yields an absolute, skip.  */
#if !defined(__clang__)
.if .-__svml_s_trig_data != \offset
.err
.endif
#endif
.rept 16
.long \value
.endr
.endm

#define __sAbsMask                      0
#define __sRangeReductionVal            64
#define __sRangeVal                     64*2
#define __sS1                           64*3
#define __sS2                           64*4
#define __sC1                           64*5
#define __sC2                           64*6
#define __sPI1                          64*7
#define __sPI2                          64*8
#define __sPI3                          64*9
#define __sPI4                          64*10
#define __sPI1_FMA                      64*11
#define __sPI2_FMA                      64*12
#define __sPI3_FMA                      64*13
#define __sA3                           64*14
#define __sA5                           64*15
#define __sA7                           64*16
#define __sA9                           64*17
#define __sA5_FMA                       64*18
#define __sA7_FMA                       64*19
#define __sA9_FMA                       64*20
#define __sInvPI                        64*21
#define __sRShifter                     64*22
#define __sHalfPI                       64*23
#define __sOneHalf                      64*24
#define __iIndexMask                  	64*25
#define __i2pK_1                      	64*26
#define __sSignMask                   	64*27
#define __dT_cosf                       64*28
#define __dT                            64*92

#endif
