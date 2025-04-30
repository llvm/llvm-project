/* Machine-dependent definitions for profiling support.  Nios II version.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#include <sysdep.h>

#define _MCOUNT_DECL(frompc, selfpc) \
static void __attribute_used__ __mcount_internal (u_long frompc, u_long selfpc)

/* This macro/func MUST save r4, r5, r6, r7 and r8 because the compiler inserts
   blind calls to mcount(), ignoring the fact that mcount may clobber
   registers; therefore, mcount may NOT clobber registers.  */

#if defined(__PIC__) || defined(PIC)
#define NIOS2_MCOUNT_CALL				      \
  "nextpc r3\n\t"					      \
  "1: movhi r2, %hiadj(_gp_got - 1b)\n\t"		      \
  "addi r2, r2, %lo(_gp_got - 1b)\n\t"			      \
  "add r2, r2, r3\n\t"						\
  "ldw r2, %call(__mcount_internal)(r2)\n\t"			\
  "callr r2\n\t"
#else
#define NIOS2_MCOUNT_CALL			\
  "call\t__mcount_internal\n\t"
#endif

#define MCOUNT						\
  asm (".globl _mcount\n\t"				\
       ".type _mcount,@function\n\t"			\
       "_mcount:\n\t"					\
       "subi sp, sp, 24\n\t"				\
       "stw ra, 20(sp)\n\t"				\
       "stw r8, 16(sp)\n\t"				\
       "stw r7, 12(sp)\n\t"				\
       "stw r6, 8(sp)\n\t"				\
       "stw r5, 4(sp)\n\t"				\
       "stw r4, 0(sp)\n\t"				\
       "mov r4, r8\n\t"					\
       "mov r5, ra\n\t"					\
       NIOS2_MCOUNT_CALL				\
       "ldw ra, 20(sp)\n\t"				\
       "ldw r8, 16(sp)\n\t"				\
       "ldw r7, 12(sp)\n\t"				\
       "ldw r6, 8(sp)\n\t"				\
       "ldw r5, 4(sp)\n\t"				\
       "ldw r4, 0(sp)\n\t"				\
       "addi sp, sp, 24\n\t"				\
       "ret\n\t"					\
       ".size _mcount, . - _mcount\n\t"			\
       );
