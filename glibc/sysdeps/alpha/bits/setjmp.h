/* Define the machine-dependent type `jmp_buf'.  Alpha version.
   Copyright (C) 1992-2021 Free Software Foundation, Inc.
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

#ifndef _BITS_SETJMP_H
#define _BITS_SETJMP_H  1

#if !defined _SETJMP_H && !defined _PTHREAD_H
# error "Never include <bits/setjmp.h> directly; use <setjmp.h> instead."
#endif

/* The previous bits/setjmp.h had __jmp_buf defined as a structure.
   We use an array of 'long int' instead, to make writing the
   assembler easier. Naturally, user code should not depend on
   either representation. */

/*
 * Integer registers:
 *    $0 is the return value (va);
 *    $1-$8, $22-$25, $28 are call-used (t0-t7, t8-t11, at);
 *    $9-$14 we save here (s0-s5);
 *    $15 is the FP and we save it here (fp or s6);
 *    $16-$21 are input arguments (call-used) (a0-a5);
 *    $26 is the return PC and we save it here (ra);
 *    $27 is the procedure value (i.e., the address of __setjmp) (pv or t12);
 *    $29 is the global pointer, which the caller will reconstruct
 *        from the return address restored in $26 (gp);
 *    $30 is the stack pointer and we save it here (sp);
 *    $31 is always zero (zero).
 *
 * Floating-point registers:
 *    $f0 is the floating return value;
 *    $f1, $f10-$f15, $f22-$f30 are call-used;
 *    $f2-$f9 we save here;
 *    $f16-$21 are input args (call-used);
 *    $f31 is always zero.
 *
 * Note that even on Alpha hardware that does not have an FPU (there
 * isn't such a thing currently) it is required to implement the FP
 * registers.
 */

#ifndef __ASSEMBLY__
typedef long int __jmp_buf[17];
#endif

#endif  /* bits/setjmp.h */
