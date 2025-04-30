/* __errno_location -- helper function for locating per-thread errno value
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#if IS_IN (rtld)
/* rtld can not access TLS too early, thus rtld_errno.

   Instead of making __open/__close pass errno from TLS to rtld_errno, simply
   use a weak __errno_location using rtld_errno, which will be overriden by the
   libc definition.  */
static int rtld_errno;
int * weak_function
__errno_location (void)
{
  return &rtld_errno;
}
libc_hidden_weak (__errno_location)
#else
#include "../../../csu/errno-loc.c"
#endif
