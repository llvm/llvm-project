/* FPU control word bits.  Alpha-mapped-to-Intel version.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Olaf Flebbe.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _ALPHA_FPU_CONTROL_H
#define _ALPHA_FPU_CONTROL_H

/*
 * Since many programs seem to hardcode the values passed to __setfpucw()
 * (rather than using the manifest constants) we emulate the x87 interface
 * here (at least where this makes sense).
 *
 *     15-13    12  11-10  9-8     7-6     5    4    3    2    1    0
 * | reserved | IC | RC  | PC | reserved | PM | UM | OM | ZM | DM | IM
 *
 * IM: Invalid operation mask
 * DM: Denormalized operand mask
 * ZM: Zero-divide mask
 * OM: Overflow mask
 * UM: Underflow mask
 * PM: Precision (inexact result) mask
 *
 * Mask bit is 1 means no interrupt.
 *
 * PC: Precision control
 * 11 - round to extended precision
 * 10 - round to double precision
 * 00 - round to single precision
 *
 * RC: Rounding control
 * 00 - rounding to nearest
 * 01 - rounding down (toward - infinity)
 * 10 - rounding up (toward + infinity)
 * 11 - rounding toward zero
 *
 * IC: Infinity control
 * That is for 8087 and 80287 only.
 *
 * The hardware default is 0x037f. I choose 0x1372.
 */

#include <features.h>

/* masking of interrupts */
#define _FPU_MASK_IM  0x01
#define _FPU_MASK_DM  0x02
#define _FPU_MASK_ZM  0x04
#define _FPU_MASK_OM  0x08
#define _FPU_MASK_UM  0x10
#define _FPU_MASK_PM  0x20

/* precision control -- without effect on Alpha */
#define _FPU_EXTENDED 0x300   /* RECOMMENDED */
#define _FPU_DOUBLE   0x200
#define _FPU_SINGLE   0x0     /* DO NOT USE */

/*
 * rounding control---notice that on the Alpha this affects only
 * instructions with the dynamic rounding mode qualifier (/d).
 */
#define _FPU_RC_NEAREST 0x000 /* RECOMMENDED */
#define _FPU_RC_DOWN    0x400
#define _FPU_RC_UP      0x800
#define _FPU_RC_ZERO    0xC00

#define _FPU_RESERVED 0xF0C0  /* Reserved bits in cw */


/* Now two recommended cw */

/* Linux default:
     - extended precision
     - rounding to positive infinity.  There is no /p instruction
       qualifier.  By setting the dynamic rounding mode to +infinity,
       one can use /d to get round to +infinity with no extra overhead
       (so long as the default isn't changed, of course...)
     - no exceptions enabled.  */

#define _FPU_DEFAULT  0x137f

/* IEEE:  same as above. */
#define _FPU_IEEE     0x137f

/* Type of the control word.  */
typedef unsigned int fpu_control_t;

/* Default control word set at startup.  */
extern fpu_control_t __fpu_control;

#endif	/* _ALPHA_FPU_CONTROL */
