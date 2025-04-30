/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>
   and Paul Janzen <pcj@primenet.com>, 1996.

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

#include <libc-lock.h>
#include <stdlib.h>
#include <utmp.h>

#include "utmp-private.h"

/* We need to protect the opening of the file.  */
__libc_lock_define_initialized (, __libc_utmp_lock attribute_hidden)


void
__setutent (void)
{
  __libc_lock_lock (__libc_utmp_lock);

  __libc_setutent ();

  __libc_lock_unlock (__libc_utmp_lock);
}
libc_hidden_def (__setutent)
weak_alias (__setutent, setutent)


int
__getutent_r (struct utmp *buffer, struct utmp **result)
{
  int retval;

  __libc_lock_lock (__libc_utmp_lock);

  retval = __libc_getutent_r (buffer, result);

  __libc_lock_unlock (__libc_utmp_lock);

  return retval;
}
libc_hidden_def (__getutent_r)
weak_alias (__getutent_r, getutent_r)


struct utmp *
__pututline (const struct utmp *data)
{
  struct utmp *buffer;

  __libc_lock_lock (__libc_utmp_lock);

  buffer = __libc_pututline (data);

  __libc_lock_unlock (__libc_utmp_lock);

  return buffer;
}
libc_hidden_def (__pututline)
weak_alias (__pututline, pututline)


void
__endutent (void)
{
  __libc_lock_lock (__libc_utmp_lock);

  __libc_endutent ();

  __libc_lock_unlock (__libc_utmp_lock);
}
libc_hidden_def (__endutent)
weak_alias (__endutent, endutent)
