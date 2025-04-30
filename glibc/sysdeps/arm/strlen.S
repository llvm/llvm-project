/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Code contributed by Matthew Wilcox <willy@odie.barnet.ac.uk>

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

/* Thumb requires excessive IT insns here.  */
#define NO_THUMB
#include <sysdep.h>

/* size_t strlen(const char *S)
 * entry: r0 -> string
 * exit: r0 = len
 */

	.syntax unified
	.text

ENTRY(strlen)
	bic     r1, r0, $3              @ addr of word containing first byte
	ldr     r2, [r1], $4            @ get the first word
	ands    r3, r0, $3              @ how many bytes are duff?
	rsb     r0, r3, $0              @ get - that number into counter.
	beq     Laligned                @ skip into main check routine if no
					@ more
#ifdef __ARMEB__
	orr     r2, r2, $0xff000000     @ set this byte to non-zero
	subs    r3, r3, $1              @ any more to do?
	orrgt   r2, r2, $0x00ff0000     @ if so, set this byte
	subs    r3, r3, $1              @ more?
	orrgt   r2, r2, $0x0000ff00     @ then set.
#else
	orr     r2, r2, $0x000000ff     @ set this byte to non-zero
	subs    r3, r3, $1              @ any more to do?
	orrgt   r2, r2, $0x0000ff00     @ if so, set this byte
	subs    r3, r3, $1              @ more?
	orrgt   r2, r2, $0x00ff0000     @ then set.
#endif
Laligned:				@ here, we have a word in r2.  Does it
	tst     r2, $0x000000ff         @ contain any zeroes?
	tstne   r2, $0x0000ff00         @
	tstne   r2, $0x00ff0000         @
	tstne   r2, $0xff000000         @
	addne   r0, r0, $4              @ if not, the string is 4 bytes longer
	ldrne   r2, [r1], $4            @ and we continue to the next word
	bne     Laligned                @
Llastword:				@ drop through to here once we find a
#ifdef __ARMEB__
	tst     r2, $0xff000000         @ word that has a zero byte in it
	addne   r0, r0, $1              @
	tstne   r2, $0x00ff0000         @ and add up to 3 bytes on to it
	addne   r0, r0, $1              @
	tstne   r2, $0x0000ff00         @ (if first three all non-zero, 4th
	addne   r0, r0, $1              @  must be zero)
#else
	tst     r2, $0x000000ff         @ word that has a zero byte in it
	addne   r0, r0, $1              @
	tstne   r2, $0x0000ff00         @ and add up to 3 bytes on to it
	addne   r0, r0, $1              @
	tstne   r2, $0x00ff0000         @ (if first three all non-zero, 4th
	addne   r0, r0, $1              @  must be zero)
#endif
	DO_RET(lr)
END(strlen)
libc_hidden_builtin_def (strlen)
