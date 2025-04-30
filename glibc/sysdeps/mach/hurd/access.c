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

#include <errno.h>
#include <unistd.h>
#include <fcntl.h>

/* Test for access to FILE by our real user and group IDs without setting
   errno.  This may be unsafe to run during initialization of tunables
   since access_common calls __hurd_file_name_lookup, which calls
   __hurd_file_name_lookup_retry, which can set errno.  */
int
__access_noerrno (const char *file, int type)
{
  return __faccessat_noerrno (AT_FDCWD, file, type, 0);
}

/* Test for access to FILE by our real user and group IDs.  */
int
__access (const char *file, int type)
{
  return __faccessat (AT_FDCWD, file, type, 0);
}
libc_hidden_def (__access)
weak_alias (__access, access)
