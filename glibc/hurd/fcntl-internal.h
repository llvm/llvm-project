/* Helper functions for translating between O_* and SOCK_* flags.
   Copyright (C) 2008-2021 Free Software Foundation, Inc.
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


#include <fcntl.h>
#include <sys/socket.h>

/* Do some compile-time checks for the SOCK_* constants, which we rely on.  */
_Static_assert (SOCK_CLOEXEC == O_CLOEXEC,
    "SOCK_CLOEXEC is assumed to be the same as O_CLOEXEC");
_Static_assert (((SOCK_MAX - 1) | SOCK_TYPE_MASK) == SOCK_TYPE_MASK,
    "SOCK_TYPE_MASK must contain SOCK_MAX - 1");
_Static_assert ((SOCK_CLOEXEC & SOCK_TYPE_MASK) == 0,
    "SOCK_TYPE_MASK must not contain SOCK_CLOEXEC");
_Static_assert ((SOCK_NONBLOCK & SOCK_TYPE_MASK) == 0,
    "SOCK_TYPE_MASK must not contain SOCK_NONBLOCK");


/* Convert from SOCK_* flags to O_* flags.  */
__extern_always_inline
int
sock_to_o_flags (int in)
{
  int out = 0;

  if (in & SOCK_NONBLOCK)
    out |= O_NONBLOCK;
  /* Others are passed through unfiltered.  */
  out |= in & ~(SOCK_NONBLOCK);

  return out;
}

/* Convert from O_* flags to SOCK_* flags.  */
__extern_always_inline
int
o_to_sock_flags (int in)
{
  int out = 0;

  if (in & O_NONBLOCK)
    out |= SOCK_NONBLOCK;
  /* Others are passed through unfiltered.  */
  out |= in & ~(O_NONBLOCK);

  return out;
}
