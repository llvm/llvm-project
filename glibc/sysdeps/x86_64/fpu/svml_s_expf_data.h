/* Offsets for data table for vector function expf.
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

#ifndef S_EXPF_DATA_H
#define S_EXPF_DATA_H

#define __sInvLn2                     	0
#define __sShifter                    	64
#define __sLn2hi                      	128
#define __sLn2lo                      	192
#define __iBias                       	256
#define __sPC0                        	320
#define __sPC1                        	384
#define __sPC2                        	448
#define __sPC3                        	512
#define __sPC4                        	576
#define __sPC5                        	640
#define __iAbsMask                    	704
#define __iDomainRange                	768

.macro float_vector offset value
/* clang integrated assembler doesn't think subtract yields an absolute, skip.  */
#if !defined(__clang__)
.if .-__svml_sexp_data != \offset
.err
.endif
#endif
.rept 16
.long \value
.endr
.endm

#endif
