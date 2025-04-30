/* Offsets for data table for function log.
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

#ifndef D_LOG_DATA_H
#define D_LOG_DATA_H

#define _LogRcp_lookup                 -4218816
#define _Log_HA_table                 	0
#define _Log_LA_table                 	8256
#define _poly_coeff_1                 	12416
#define _poly_coeff_2                 	12480
#define _poly_coeff_3                 	12544
#define _poly_coeff_4                 	12608
#define _ExpMask                      	12672
#define _Two10                        	12736
#define _MinNorm                      	12800
#define _MaxNorm                      	12864
#define _HalfMask                     	12928
#define _One                          	12992
#define _L2H                          	13056
#define _L2L                          	13120
#define _Threshold                    	13184
#define _Bias                         	13248
#define _Bias1                        	13312
#define _L2                           	13376
#define _dInfs                        	13440
#define _dOnes                        	13504
#define _dZeros                       	13568

.macro double_vector offset value
/* clang integrated assembler doesn't think subtract yields an absolute, skip.  */
#if !defined(__clang__)
.if .-__svml_dlog_data != \offset
.err
.endif
#endif
.rept 8
.quad \value
.endr
.endm

#endif
