/* lround function.  POWER6x, PowerPC32 version.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
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

#include <sysdep.h>
#include <math_ldbl_opt.h>
#include <libm-alias-float.h>
#include <libm-alias-double.h>

/* long [r3] lround (float x [fp1])
   IEEE 1003.1 lround function.  IEEE specifies "round to the nearest
   integer value, rounding halfway cases away from zero, regardless of
   the current rounding mode."  However PowerPC Architecture defines
   "round to Nearest" as "Choose the best approximation. In case of a
   tie, choose the one that is even (least significant bit o).".
   So we pre-round using the V2.02 Floating Round to Integer Nearest
   instruction before we use the Floating Convert to Integer Word with
   round to zero instruction.  */

	.machine	"power6"
ENTRY (__lround)
	frin	fp2,fp1	/* Pre-round +-0.5.  */
	fctiwz	fp3,fp2	/* Convert To Integer Word lround toward 0.  */
	mftgpr	r3,fp3	/* Transfer fpr3 to r3.  */
	blr
	END (__lround)

libm_alias_double (__lround, lround)

strong_alias (__lround, __lroundf)
libm_alias_float (__lround, lround)
