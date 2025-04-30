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

#include <fcntl.h>
#include <unistd.h>
#include <paths.h>
#include <stdlib.h>

/* Path to the master pseudo terminal cloning device.  */
#define _PATH_DEVPTMX _PATH_DEV "ptmx"

/* Open a master pseudo terminal and return its file descriptor.  */
int
__posix_openpt (int oflag)
{
  return __open (_PATH_DEVPTMX, oflag);
}
weak_alias (__posix_openpt, posix_openpt)


int
__getpt (void)
{
  return __posix_openpt (O_RDWR);
}
libc_hidden_def (__getpt)
weak_alias (__getpt, getpt)
