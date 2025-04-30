/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Brendan Kehoe (brendan@zen.org).

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

#include <setjmp.h>

/* This function is only called via the assembly language routine
   __sigsetjmp, which arranges to pass in the stack pointer and the frame
   pointer.  We do things this way because it's difficult to reliably
   access them in C.  */

/* Stack protection is disabled to avoid changing s0 (or any other
   caller-save register) before storing it to environment.
   See BZ #22624.  */

int __attribute__ ((nomips16))
inhibit_stack_protector
__sigsetjmp_aux (jmp_buf env, int savemask, int sp, int fp)
{
#ifdef __mips_hard_float
  /* Store the floating point callee-saved registers...  */
  asm volatile ("s.d $f20, %0" : : "m" (env[0].__jmpbuf[0].__fpregs[0]));
  asm volatile ("s.d $f22, %0" : : "m" (env[0].__jmpbuf[0].__fpregs[1]));
  asm volatile ("s.d $f24, %0" : : "m" (env[0].__jmpbuf[0].__fpregs[2]));
  asm volatile ("s.d $f26, %0" : : "m" (env[0].__jmpbuf[0].__fpregs[3]));
  asm volatile ("s.d $f28, %0" : : "m" (env[0].__jmpbuf[0].__fpregs[4]));
  asm volatile ("s.d $f30, %0" : : "m" (env[0].__jmpbuf[0].__fpregs[5]));
#endif

  /* .. and the PC;  */
  asm volatile ("sw $31, %0" : : "m" (env[0].__jmpbuf[0].__pc));

  /* .. and the stack pointer;  */
  env[0].__jmpbuf[0].__sp = (void *) sp;

  /* .. and the FP; it'll be in s8. */
  env[0].__jmpbuf[0].__fp = (void *) fp;

  /* .. and the GP; */
  asm volatile ("sw $gp, %0" : : "m" (env[0].__jmpbuf[0].__gp));

  /* .. and the callee-saved registers; */
  asm volatile ("sw $16, %0" : : "m" (env[0].__jmpbuf[0].__regs[0]));
  asm volatile ("sw $17, %0" : : "m" (env[0].__jmpbuf[0].__regs[1]));
  asm volatile ("sw $18, %0" : : "m" (env[0].__jmpbuf[0].__regs[2]));
  asm volatile ("sw $19, %0" : : "m" (env[0].__jmpbuf[0].__regs[3]));
  asm volatile ("sw $20, %0" : : "m" (env[0].__jmpbuf[0].__regs[4]));
  asm volatile ("sw $21, %0" : : "m" (env[0].__jmpbuf[0].__regs[5]));
  asm volatile ("sw $22, %0" : : "m" (env[0].__jmpbuf[0].__regs[6]));
  asm volatile ("sw $23, %0" : : "m" (env[0].__jmpbuf[0].__regs[7]));

  /* Save the signal mask if requested.  */
  return __sigjmp_save (env, savemask);
}
