/* Offsets for data table for vectorized sin, cos, sincos.
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

#ifndef D_TRIG_DATA_H
#define D_TRIG_DATA_H

#define __dAbsMask              0
#define __dRangeVal             64
#define __dRangeVal_sin         64*2
#define __dHalfPI               64*3
#define __dInvPI                64*4
#define __dRShifter             64*5
#define __dZero                 64*6
#define __lNZero                64*7
#define __dOneHalf              64*8
#define __dPI1                  64*9
#define __dPI2                  64*10
#define __dPI3                  64*11
#define __dPI4                  64*12
#define __dPI1_FMA              64*13
#define __dPI2_FMA              64*14
#define __dPI3_FMA              64*15
#define __dHalfPI1              64*16
#define __dHalfPI2              64*17
#define __dHalfPI3              64*18
#define __dHalfPI4              64*19
#define __dC1                   64*20
#define __dC2                   64*21
#define __dC3                   64*22
#define __dC4                   64*23
#define __dC5                   64*24
#define __dC6                   64*25
#define __dC7                   64*26
#define __dC1_sin               64*27
#define __dC2_sin               64*28
#define __dC3_sin               64*29
#define __dC4_sin               64*30
#define __dC5_sin               64*31
#define __dC6_sin               64*32
#define __dC7_sin               64*33
#define __dRShifter_la          64*34
#define __dRShifterm5_la        64*35
#define __dRXmax_la             64*36
#define __dAbsMask_la           __dAbsMask
#define __dInvPI_la             __dInvPI
#define __dSignMask             __lNZero

.macro double_vector offset value
/* clang integrated assembler doesn't think subtract yields an absolute, skip.  */
#if !defined(__clang__)
.if .-__svml_d_trig_data != \offset
.err
.endif
#endif
.rept 8
.quad \value
.endr
.endm

#endif
