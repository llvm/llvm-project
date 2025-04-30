/* The mcheck() interface.
   Copyright (C) 1990-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written May 1989 by Mike Haertel.

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

#if !IS_IN (libc)
# include "mcheck-impl.c"
#else
# include <mcheck.h>
#endif

void
mcheck_check_all (void)
{
#if !IS_IN (libc)
  __mcheck_checkptr (NULL);
#endif
}

int
mcheck (void (*func) (enum mcheck_status))
{
#if IS_IN (libc)
  return -1;
#else
  return __mcheck_initialize (func, false);
#endif
}

int
mcheck_pedantic (void (*func) (enum mcheck_status))
{
#if IS_IN (libc)
  return -1;
#else
  return __mcheck_initialize (func, true);
#endif
}

enum mcheck_status
mprobe (void *ptr)
{
#if IS_IN (libc)
  return MCHECK_DISABLED;
#else
  return __mcheck_checkptr (ptr);
#endif
}
