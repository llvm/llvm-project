/* Internal definitions and declarations for UTMP functions.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#ifndef _UTMP_PRIVATE_H
#define _UTMP_PRIVATE_H	1

#include <utmp.h>
#include <libc-lock.h>

/* These functions check for initialization, but not perform any
   locking.  */
int __libc_setutent (void) attribute_hidden;
int __libc_getutent_r (struct utmp *, struct utmp **) attribute_hidden;
int __libc_getutid_r (const struct utmp *, struct utmp *, struct utmp **)
  attribute_hidden;
int __libc_getutline_r (const struct utmp *, struct utmp *, struct utmp **)
  attribute_hidden;
struct utmp *__libc_pututline (const struct utmp *) attribute_hidden;
void __libc_endutent (void) attribute_hidden;
int __libc_updwtmp (const char *, const struct utmp *) attribute_hidden;

/* Current file name.  */
extern const char *__libc_utmp_file_name attribute_hidden;

/* Locks access to the global data.  */
__libc_lock_define (extern, __libc_utmp_lock attribute_hidden)


#endif /* utmp-private.h */
