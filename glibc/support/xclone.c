/* Auxiliary functions to issue the clone syscall.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#ifdef __linux__
# include <support/check.h>
# include <stackinfo.h>  /* For _STACK_GROWS_{UP,DOWN}.  */
# include <xsched.h>

pid_t
xclone (int (*fn) (void *arg), void *arg, void *stack, size_t stack_size,
	int flags)
{
  pid_t r = -1;

# ifdef __ia64__
  extern int __clone2 (int (*fn) (void *arg), void *stack, size_t stack_size,
		       int flags, void *arg, ...);
  r = __clone2 (fn, stack, stack_size, flags, arg, /* ptid */ NULL,
		/* tls */ NULL, /* ctid  */ NULL);
# else
#  if _STACK_GROWS_DOWN
  r = clone (fn, stack + stack_size, flags, arg, /* ptid */ NULL,
	     /* tls */ NULL, /* ctid */  NULL);
#  elif _STACK_GROWS_UP
  r = clone (fn, stack, flags, arg, /* ptid */ NULL, /* tls */ NULL, NULL);
#  endif
# endif

  if (r < 0)
    FAIL_EXIT1 ("clone: %m");

  return r;
}
#endif
