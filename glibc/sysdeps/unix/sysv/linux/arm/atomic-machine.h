/* Atomic operations.  ARM/Linux version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#include <stdint.h>

/* If the compiler doesn't provide a primitive, we'll use this macro
   to get assistance from the kernel.  */
#ifdef __thumb2__
# define __arm_assisted_full_barrier() \
     __asm__ __volatile__						      \
	     ("movw\tip, #0x0fa0\n\t"					      \
	      "movt\tip, #0xffff\n\t"					      \
	      "blx\tip"							      \
	      : : : "ip", "lr", "cc", "memory");
#else
# define __arm_assisted_full_barrier() \
     __asm__ __volatile__						      \
	     ("mov\tip, #0xffff0fff\n\t"				      \
	      "mov\tlr, pc\n\t"						      \
	      "add\tpc, ip, #(0xffff0fa0 - 0xffff0fff)"			      \
	      : : : "ip", "lr", "cc", "memory");
#endif

/* Atomic compare and exchange.  This sequence relies on the kernel to
   provide a compare and exchange operation which is atomic on the
   current architecture, either via cleverness on pre-ARMv6 or via
   ldrex / strex on ARMv6.

   It doesn't matter what register is used for a_oldval2, but we must
   specify one to work around GCC PR rtl-optimization/21223.  Otherwise
   it may cause a_oldval or a_tmp to be moved to a different register.

   We use the union trick rather than simply using __typeof (...) in the
   declarations of A_OLDVAL et al because when NEWVAL or OLDVAL is of the
   form *PTR and PTR has a 'volatile ... *' type, then __typeof (*PTR) has
   a 'volatile ...' type and this triggers -Wvolatile-register-var to
   complain about 'register volatile ... asm ("reg")'.

   We use the same union trick in the declaration of A_PTR because when
   MEM is of the from *PTR and PTR has a 'const ... *' type, then __typeof
   (*PTR) has a 'const ...' type and this enables the compiler to substitute
   the variable with its initializer in asm statements, which may cause the
   corresponding operand to appear in a different register.  */
#ifdef __thumb2__
/* Thumb-2 has ldrex/strex.  However it does not have barrier instructions,
   so we still need to use the kernel helper.  */
# define __arm_assisted_compare_and_exchange_val_32_acq(mem, newval, oldval) \
  ({ union { __typeof (mem) a; uint32_t v; } mem_arg = { .a = (mem) };       \
     union { __typeof (oldval) a; uint32_t v; } oldval_arg = { .a = (oldval) };\
     union { __typeof (newval) a; uint32_t v; } newval_arg = { .a = (newval) };\
     register uint32_t a_oldval asm ("r0");				      \
     register uint32_t a_newval asm ("r1") = newval_arg.v;		      \
     register uint32_t a_ptr asm ("r2") = mem_arg.v;			      \
     register uint32_t a_tmp asm ("r3");				      \
     register uint32_t a_oldval2 asm ("r4") = oldval_arg.v;		      \
     __asm__ __volatile__						      \
	     ("0:\tldr\t%[tmp],[%[ptr]]\n\t"				      \
	      "cmp\t%[tmp], %[old2]\n\t"				      \
	      "bne\t1f\n\t"						      \
	      "mov\t%[old], %[old2]\n\t"				      \
	      "movw\t%[tmp], #0x0fc0\n\t"				      \
	      "movt\t%[tmp], #0xffff\n\t"				      \
	      "blx\t%[tmp]\n\t"						      \
	      "bcc\t0b\n\t"						      \
	      "mov\t%[tmp], %[old2]\n\t"				      \
	      "1:"							      \
	      : [old] "=&r" (a_oldval), [tmp] "=&r" (a_tmp)		      \
	      : [new] "r" (a_newval), [ptr] "r" (a_ptr),		      \
		[old2] "r" (a_oldval2)					      \
	      : "ip", "lr", "cc", "memory");				      \
     (__typeof (oldval)) a_tmp; })
#else
# define __arm_assisted_compare_and_exchange_val_32_acq(mem, newval, oldval) \
  ({ union { __typeof (mem) a; uint32_t v; } mem_arg = { .a = (mem) };       \
     union { __typeof (oldval) a; uint32_t v; } oldval_arg = { .a = (oldval) };\
     union { __typeof (newval) a; uint32_t v; } newval_arg = { .a = (newval) };\
     register uint32_t a_oldval asm ("r0");				      \
     register uint32_t a_newval asm ("r1") = newval_arg.v;		      \
     register uint32_t a_ptr asm ("r2") = mem_arg.v;			      \
     register uint32_t a_tmp asm ("r3");				      \
     register uint32_t a_oldval2 asm ("r4") = oldval_arg.v;		      \
     __asm__ __volatile__						      \
	     ("0:\tldr\t%[tmp],[%[ptr]]\n\t"				      \
	      "cmp\t%[tmp], %[old2]\n\t"				      \
	      "bne\t1f\n\t"						      \
	      "mov\t%[old], %[old2]\n\t"				      \
	      "mov\t%[tmp], #0xffff0fff\n\t"				      \
	      "mov\tlr, pc\n\t"						      \
	      "add\tpc, %[tmp], #(0xffff0fc0 - 0xffff0fff)\n\t"		      \
	      "bcc\t0b\n\t"						      \
	      "mov\t%[tmp], %[old2]\n\t"				      \
	      "1:"							      \
	      : [old] "=&r" (a_oldval), [tmp] "=&r" (a_tmp)		      \
	      : [new] "r" (a_newval), [ptr] "r" (a_ptr),		      \
		[old2] "r" (a_oldval2)					      \
	      : "ip", "lr", "cc", "memory");				      \
     (__typeof (oldval)) a_tmp; })
#endif

#include <sysdeps/arm/atomic-machine.h>
