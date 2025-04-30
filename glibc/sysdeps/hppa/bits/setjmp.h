/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

/* Define the machine-dependent type `jmp_buf'.  HPPA version.  */
#ifndef _BITS_SETJMP_H
#define _BITS_SETJMP_H	1

#if !defined _SETJMP_H && !defined _PTHREAD_H
# error "Never include <bits/setjmp.h> directly; use <setjmp.h> instead."
#endif

#ifndef	_ASM
/* The entire jump buffer must be 168 bytes long and laid
   out in exactly as follows for ABI consistency.
   * 20 x 32-bit gprs, with 8-bytes of padding, arranged so:
     - r3 (callee saves)
     - 4 bytes of padding.
     - r4-r18 (callee saves)
     - r19 (PIC register)
     - r27 (static link register)
     - r30 (stcack pointer)
     - r2 (return pointer)
     - 4 bytes of padding.
   * 10 x 64-bit fprs in this order:
     - fr12-fr21 (callee saves)
   Note: We have 8 bytes of free space for future uses.  */
typedef union __jmp_buf_internal_tag
  {
    struct
      {
	int __r3;
	int __pad0;
	int __r4_r18[15];
	int __r19;
	int __r27;
	int __sp;
	int __rp;
	int __pad1;
	double __fr12_fr21[10];
      } __jmp_buf;
    /* Legacy definition. Ensures double alignment for fpsrs.  */
    double __align[21];
  } __jmp_buf[1];
#endif

#endif	/* bits/setjmp.h */
