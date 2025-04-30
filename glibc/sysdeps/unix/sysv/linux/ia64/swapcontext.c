/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
     Contributed by David Mosberger-Tang <davidm@hpl.hp.com>.

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

struct rv
  {
    long retval;
    long first_return;
  };

extern struct rv __getcontext (ucontext_t *__ucp) __THROW;
extern int __setcontext (const ucontext_t *__ucp) __THROW;

int
__swapcontext (ucontext_t *oucp, const ucontext_t *ucp)
{
  struct rv rv = __getcontext (oucp);
  if (rv.first_return)
    __setcontext (ucp);
  return 0;
}

weak_alias (__swapcontext, swapcontext)
