/* FPU control word definitions.  SH version.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
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

#ifndef _FPU_CONTROL_H
#define _FPU_CONTROL_H

#if !defined(__SH_FPU_ANY__)

#define _FPU_RESERVED 0xffffffff
#define _FPU_DEFAULT  0x00000000
typedef unsigned int fpu_control_t;
#define _FPU_GETCW(cw) (cw) = 0
#define _FPU_SETCW(cw) (void) (cw)
extern fpu_control_t __fpu_control;

#else

#include <features.h>

/* masking of interrupts */
#define _FPU_MASK_VM	0x0800	/* Invalid operation */
#define _FPU_MASK_ZM	0x0400	/* Division by zero  */
#define _FPU_MASK_OM	0x0200	/* Overflow	     */
#define _FPU_MASK_UM	0x0100	/* Underflow	     */
#define _FPU_MASK_IM	0x0080	/* Inexact operation */

/* rounding control */
#define _FPU_RC_NEAREST 0x0	/* RECOMMENDED */
#define _FPU_RC_ZERO	0x1

#define _FPU_RESERVED 0xffc00000  /* These bits are reserved.  */

/* The fdlibm code requires strict IEEE double precision arithmetic,
   and no interrupts for exceptions, rounding to nearest.  */
#define _FPU_DEFAULT	0x00080000 /* Default value.  */
#define _FPU_IEEE	0x00080f80 /* Default + exceptions enabled. */

/* Type of the control word.  */
typedef unsigned int fpu_control_t;

/* Macros for accessing the hardware control word.  */
#define _FPU_GETCW(cw) __asm__ ("sts fpscr,%0" : "=r" (cw))

#if defined __GNUC__
__BEGIN_DECLS

/* GCC provides this function.  */
extern void __set_fpscr (unsigned long);
#define _FPU_SETCW(cw) __set_fpscr ((cw))
#else
#define _FPU_SETCW(cw) __asm__ ("lds %0,fpscr" : : "r" (cw))
#endif

/* Default control word set at startup.	 */
extern fpu_control_t __fpu_control;

__END_DECLS

#endif /* __SH_FPU_ANY__ */

#endif /* _FPU_CONTROL_H */
