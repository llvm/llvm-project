/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

/* Mark symbols hidden in static PIE for early self relocation to work.  */
#if BUILD_PIE_DEFAULT
# pragma GCC visibility push(hidden)
#endif
#include <errno.h>
#include <libc-internal.h>
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>

/* Defined in brk.c.  */
extern void *__curbrk;
extern int __brk (void *addr);

/* Extend the process's data space by INCREMENT.
   If INCREMENT is negative, shrink data space by - INCREMENT.
   Return start of new space allocated, or -1 for errors.  */
void *
__sbrk (intptr_t increment)
{
  /* Controls whether __brk (0) is called to read the brk value from
     the kernel.  */
  bool update_brk = __curbrk == NULL;

#if defined (SHARED) && ! IS_IN (rtld)
  if (!__libc_initial)
    {
      if (increment != 0)
	{
	  /* Do not allow changing the brk from an inner libc because
	     it cannot be synchronized with the outer libc's brk.  */
	  __set_errno (ENOMEM);
	  return (void *) -1;
	}
      /* Querying the kernel's brk value from an inner namespace is
	 fine.  */
      update_brk = true;
    }
#endif

  if (update_brk)
    if (__brk (0) < 0)		/* Initialize the break.  */
      return (void *) -1;

  if (increment == 0)
    return __curbrk;

  void *oldbrk = __curbrk;
  if (increment > 0
      ? ((uintptr_t) oldbrk + (uintptr_t) increment < (uintptr_t) oldbrk)
      : ((uintptr_t) oldbrk < (uintptr_t) -increment))
    {
      __set_errno (ENOMEM);
      return (void *) -1;
    }

  if (__brk (oldbrk + increment) < 0)
    return (void *) -1;

  return oldbrk;
}
libc_hidden_def (__sbrk)
weak_alias (__sbrk, sbrk)
