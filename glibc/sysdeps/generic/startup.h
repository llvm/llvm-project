/* Generic definitions of functions used by static libc main startup.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

/* Targets should override this file if the default definitions below
   will not work correctly very early before TLS is initialized.  */

#include <unistd.h>

/* Use macro instead of inline function to avoid including <stdio.h>.  */
#define _startup_fatal(message) __libc_fatal ((message))

static inline uid_t
startup_getuid (void)
{
  return __getuid ();
}

static inline uid_t
startup_geteuid (void)
{
  return __geteuid ();
}

static inline gid_t
startup_getgid (void)
{
  return __getgid ();
}

static inline gid_t
startup_getegid (void)
{
  return __getegid ();
}
