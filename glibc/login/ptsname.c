/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Zack Weinberg <zack@rabi.phys.columbia.edu>, 1998.

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

#include <errno.h>
#include <stdlib.h>
#include <unistd.h>

/* Given the file descriptor of a master pty, return the pathname
   of the associated slave. */

char *
ptsname (int fd __attribute__ ((unused)))
{
  __set_errno (ENOSYS);
  return NULL;
}

int
__ptsname_r (int fd __attribute__ ((unused)),
	     char *buf __attribute__ ((unused)),
	     size_t len __attribute__ ((unused)))
{
  __set_errno (ENOSYS);
  return ENOSYS;
}
libc_hidden_def (__ptsname_r)
weak_alias (__ptsname_r, ptsname_r)

stub_warning(ptsname)
stub_warning(ptsname_r)
