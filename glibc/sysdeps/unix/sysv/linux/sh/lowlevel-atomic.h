/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
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

#ifdef __ASSEMBLER__

#define _IMP1 #1
#define _IMM1 #-1
#define _IMM4 #-4
#define _IMM6 #-6
#define _IMM8 #-8

#define	INC(mem, reg) \
	.align	2; \
	mova	99f, r0; \
	mov	r15, r1; \
	mov	_IMM6, r15; \
98:	mov.l	mem, reg; \
	add	_IMP1, reg; \
	mov.l	reg, mem; \
99:	mov	r1, r15

#define	DEC(mem, reg) \
	.align	2; \
	mova	99f, r0; \
	mov	r15, r1; \
	mov	_IMM6, r15; \
98:	mov.l	mem, reg; \
	add	_IMM1, reg; \
	mov.l	reg, mem; \
99:	mov	r1, r15

#define	XADD(reg, mem, old, tmp) \
	.align	2; \
	mova	99f, r0; \
	nop; \
	mov	r15, r1; \
	mov	_IMM8, r15; \
98:	mov.l	mem, old; \
	mov	reg, tmp; \
	add	old, tmp; \
	mov.l	tmp, mem; \
99:	mov	r1, r15

#define	XCHG(reg, mem, old) \
	.align	2; \
	mova	99f, r0; \
	nop; \
	mov	r15, r1; \
	mov	_IMM4, r15; \
98:	mov.l	mem, old; \
	mov.l	reg, mem; \
99:	mov	r1, r15

#define	CMPXCHG(reg, mem, new, old) \
	.align	2; \
	mova	99f, r0; \
	nop; \
	mov	r15, r1; \
	mov	_IMM8, r15; \
98:	mov.l	mem, old; \
	cmp/eq	old, reg; \
	bf	99f; \
	mov.l	new, mem; \
99:	mov	r1, r15

#endif  /* __ASSEMBLER__ */
