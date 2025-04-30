/* FPU control word definitions.  HP-PARISC version.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _FPU_CONTROL_H
#define _FPU_CONTROL_H

/* Masking of interrupts.  */
#define _FPU_MASK_PM	0x00000001	/* Inexact (I) */
#define _FPU_MASK_UM	0x00000002	/* Underflow (U) */
#define _FPU_MASK_OM	0x00000004	/* Overflow (O) */
#define _FPU_MASK_ZM	0x00000008	/* Divide by zero (Z) */
#define _FPU_MASK_IM	0x00000010	/* Invalid operation (V) */

/* Masking of rounding modes.  */
#define _FPU_HPPA_MASK_RM	0x00000600	/* Rounding mode mask */
/* Masking of interrupt enable bits.  */
#define _FPU_HPPA_MASK_INT	0x0000001f	/* Interrupt mask */
/* Shift by 27 to install flag bits.  */
#define _FPU_HPPA_SHIFT_FLAGS	27

/* There are no reserved bits in the PA fpsr (though some are undefined).  */
#define _FPU_RESERVED	0x00000000
/* Default is: No traps enabled, no flags set, round to nearest.  */
#define _FPU_DEFAULT    0x00000000
/* Default + exceptions (FE_ALL_EXCEPT) enabled. */
#define _FPU_IEEE	(_FPU_DEFAULT | _FPU_HPPA_MASK_INT)

/* Type of the control word.  */
typedef unsigned int fpu_control_t;

/* Macros for accessing the hardware control word.  */
#define _FPU_GETCW(cw) \
({										\
  union { __extension__ unsigned long long __fpreg; unsigned int __halfreg[2]; } __fullfp; \
  /* Get the current status word. */						\
  __asm__ ("fstd %%fr0,0(%1)\n\t"						\
           "fldd 0(%1),%%fr0\n\t"						\
	   : "=m" (__fullfp.__fpreg) : "r" (&__fullfp.__fpreg) : "%r0");	\
  cw = __fullfp.__halfreg[0];							\
})

#define _FPU_SETCW(cw) \
({										\
  union { __extension__ unsigned long long __fpreg; unsigned int __halfreg[2]; } __fullfp;	\
  /* Get the current status word and set the control word.  */			\
  __asm__ ("fstd %%fr0,0(%1)\n\t"						\
	   : "=m" (__fullfp.__fpreg) : "r" (&__fullfp.__fpreg) : "%r0");	\
  __fullfp.__halfreg[0] = cw;							\
  __asm__ ("fldd 0(%1),%%fr0\n\t"						\
	   : : "m" (__fullfp.__fpreg), "r" (&__fullfp.__fpreg) : "%r0" );	\
})

/* Default control word set at startup.  */
extern fpu_control_t __fpu_control;

#endif /* _FPU_CONTROL_H */
