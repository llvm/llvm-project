/* setjmp for RISC-V.
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
#include <sys/asm.h>

ENTRY (_setjmp)
  li	a1, 0
  j	__sigsetjmp
END (_setjmp)
ENTRY (setjmp)
  li	a1, 1
  /* Fallthrough */
END (setjmp)
ENTRY (__sigsetjmp)
	REG_S ra,  0*SZREG(a0)
	REG_S s0,  1*SZREG(a0)
	REG_S s1,  2*SZREG(a0)
	REG_S s2,  3*SZREG(a0)
	REG_S s3,  4*SZREG(a0)
	REG_S s4,  5*SZREG(a0)
	REG_S s5,  6*SZREG(a0)
	REG_S s6,  7*SZREG(a0)
	REG_S s7,  8*SZREG(a0)
	REG_S s8,  9*SZREG(a0)
	REG_S s9, 10*SZREG(a0)
	REG_S s10,11*SZREG(a0)
	REG_S s11,12*SZREG(a0)
	REG_S sp, 13*SZREG(a0)

#ifndef __riscv_float_abi_soft
	FREG_S fs0, 14*SZREG+ 0*SZFREG(a0)
	FREG_S fs1, 14*SZREG+ 1*SZFREG(a0)
	FREG_S fs2, 14*SZREG+ 2*SZFREG(a0)
	FREG_S fs3, 14*SZREG+ 3*SZFREG(a0)
	FREG_S fs4, 14*SZREG+ 4*SZFREG(a0)
	FREG_S fs5, 14*SZREG+ 5*SZFREG(a0)
	FREG_S fs6, 14*SZREG+ 6*SZFREG(a0)
	FREG_S fs7, 14*SZREG+ 7*SZFREG(a0)
	FREG_S fs8, 14*SZREG+ 8*SZFREG(a0)
	FREG_S fs9, 14*SZREG+ 9*SZFREG(a0)
	FREG_S fs10,14*SZREG+10*SZFREG(a0)
	FREG_S fs11,14*SZREG+11*SZFREG(a0)
#endif

#if !IS_IN (libc) && IS_IN (rtld)
  /* In ld.so we never save the signal mask.  */
  li a0, 0
  ret
#else
  /* Make a tail call to __sigjmp_save; it takes the same args.  */
  j __sigjmp_save
#endif


END (__sigsetjmp)

hidden_def (__sigsetjmp)
weak_alias (_setjmp, __GI__setjmp)
