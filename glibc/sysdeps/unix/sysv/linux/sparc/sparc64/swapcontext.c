/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>.

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

#include <ucontext.h>

extern int __getcontext (ucontext_t *ucp);
extern int __setcontext (const ucontext_t *ucp, int restoremask);

int
__swapcontext (ucontext_t *oucp, const ucontext_t *ucp)
{
  extern void __swapcontext_ret (void);
  /* Save the current machine context to oucp.  */
  __getcontext (oucp);
  /* Modify oucp to skip the __setcontext call on reactivation.  */
  oucp->uc_mcontext.mc_gregs[MC_PC] = (long) __swapcontext_ret;
  oucp->uc_mcontext.mc_gregs[MC_NPC] = ((long) __swapcontext_ret) + 4;
  /* Restore the machine context in ucp.  */
  __setcontext (ucp, 1);
  return 0;
}

asm ("							\n\
	.text						\n\
	.type	__swapcontext_ret, #function		\n\
__swapcontext_ret:					\n\
	return	%i7 + 8					\n\
	 clr	%o0					\n\
	.size	__swapcontext_ret, .-__swapcontext_ret	\n\
     ");

weak_alias (__swapcontext, swapcontext)
