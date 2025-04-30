/* Offsets for data table for vectorized function logf.
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

#ifndef S_LOGF_DATA_H
#define S_LOGF_DATA_H

#define _sPoly_1                      	0
#define _sPoly_2                      	64
#define _sPoly_3                      	128
#define _sPoly_4                      	192
#define _sPoly_5                      	256
#define _sPoly_6                      	320
#define _sPoly_7                      	384
#define _iHiDelta                     	448
#define _iLoRange                     	512
#define _iBrkValue                    	576
#define _iOffExpoMask                 	640
#define _sOne                         	704
#define _sLn2                         	768
#define _sInfs                        	832
#define _sOnes                        	896
#define _sZeros                       	960

.macro float_vector offset value
/* clang integrated assembler doesn't think subtract yields an absolute, skip.  */
#if !defined(__clang__)
.if .-__svml_slog_data != \offset
.err
.endif
#endif
.rept 16
.long \value
.endr
.endm

#endif
