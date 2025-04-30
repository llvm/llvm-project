/* Offsets for data table for function powf.
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

#ifndef S_POWF_DATA_H
#define S_POWF_DATA_H

#define _Log2Rcp_lookup                -4218496
#define _NMINNORM                     	0
#define _NMAXVAL                      	64
#define _INF                          	128
#define _ABSMASK                      	192
#define _DOMAINRANGE                  	256
#define _Log_HA_table                 	320
#define _Log_LA_table                 	8576
#define _poly_coeff_1                 	12736
#define _poly_coeff_2                 	12800
#define _poly_coeff_3                 	12864
#define _poly_coeff_4                 	12928
#define _ExpMask                      	12992
#define _Two10                        	13056
#define _MinNorm                      	13120
#define _MaxNorm                      	13184
#define _HalfMask                     	13248
#define _One                          	13312
#define _L2H                          	13376
#define _L2L                          	13440
#define _Threshold                    	13504
#define _Bias                         	13568
#define _Bias1                        	13632
#define _L2                           	13696
#define _dInfs                        	13760
#define _dOnes                        	13824
#define _dZeros                       	13888
#define __dbT                         	13952
#define __dbInvLn2                    	30400
#define __dbShifter                   	30464
#define __dbHALF                      	30528
#define __dbC1                        	30592
#define __lbLOWKBITS                  	30656
#define __iAbsMask                    	30720
#define __iDomainRange                	30784

.macro double_vector offset value
/* clang integrated assembler doesn't think subtract yields an absolute, skip.  */
#if !defined(__clang__)
.if .-__svml_spow_data != \offset
.err
.endif
#endif
.rept 8
.quad \value
.endr
.endm

.macro float_vector offset value
/* clang integrated assembler doesn't think subtract yields an absolute, skip.  */
#if !defined(__clang__)
.if .-__svml_spow_data != \offset
.err
.endif
#endif
.rept 16
.long \value
.endr
.endm

#endif
