/* Swap to new context.
   Copyright (C) 2008-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Helge Deller <deller@gmx.de>, 2008.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <ucontext.h>

extern int __getcontext (ucontext_t *ucp);
extern int __setcontext (const ucontext_t *ucp);

int
__swapcontext (ucontext_t *oucp, const ucontext_t *ucp)
{
  /* Save the current machine context to oucp.  */
  __getcontext (oucp);

  /* mark sc_sar flag to skip the setcontext call on reactivation.  */
  if (oucp->uc_mcontext.sc_sar == 0) {
	oucp->uc_mcontext.sc_sar++;

	/* Restore the machine context in ucp.  */
	__setcontext (ucp);
  }

  return 0;
}

weak_alias (__swapcontext, swapcontext)
