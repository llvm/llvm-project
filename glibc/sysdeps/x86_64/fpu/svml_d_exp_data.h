/* Offsets for data table for function exp.
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

#ifndef D_EXP_DATA_H
#define D_EXP_DATA_H

#define __dbT                         	0
#define __dbInvLn2                    	8192
#define __dbShifter                   	8256
#define __dbLn2hi                     	8320
#define __dbLn2lo                     	8384
#define __dPC0                        	8448
#define __dPC1                        	8512
#define __dPC2                        	8576
#define __lIndexMask                  	8640
#define __iAbsMask                    	8704
#define __iDomainRange                	8768

.macro double_vector offset value
/* clang integrated assembler doesn't think subtract yields an absolute, skip.  */
#if !defined(__clang__)
.if .-__svml_dexp_data != \offset
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
.if .-__svml_dexp_data != \offset
.err
.endif
#endif
.rept 16
.long \value
.endr
.endm

#endif
