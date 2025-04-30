/* Linux/alpha select implementation.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sys/time.h>
#include <sys/types.h>
#include <sys/select.h>
#include <errno.h>
#include <sysdep-cancel.h>
#include <shlib-compat.h>

int
__new_select (int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds,
	      struct timeval *timeout)
{
  return SYSCALL_CANCEL (select, nfds, readfds, writefds, exceptfds, timeout);
}
strong_alias (__new_select, __select)
libc_hidden_def (__select)

default_symbol_version (__new_select, select, GLIBC_2.1);

strong_alias (__new_select, __new_select_private);
symbol_version (__new_select_private, __select, GLIBC_2.1);

/* Old timeval32 compat calls.  */
#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_1)
int
__select_tv32 (int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds,
	       struct timeval *timeout)
{
  return SYSCALL_CANCEL (osf_select, nfds, readfds, writefds, exceptfds,
                        timeout);
}
strong_alias (__select_tv32, __select_tv32_1)

compat_symbol (libc, __select_tv32, __select, GLIBC_2_0);
compat_symbol (libc, __select_tv32_1, select, GLIBC_2_0);
#endif
