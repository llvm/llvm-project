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

#define sigpause __rename_sigpause
#include <errno.h>
#include <signal.h>
#undef sigpause

int
__sigpause (int sig_or_mask, int is_sig)
{
  __set_errno (ENOSYS);
  return -1;
}
stub_warning (__sigpause)
libc_hidden_def (__sigpause)

int
__attribute__ ((weak))
__default_sigpause (int mask)
{
  __set_errno (ENOSYS);
  return -1;
}
weak_alias (__default_sigpause, sigpause)
stub_warning (sigpause)


int
__attribute ((weak))
__xpg___sigpause (int sig)
{
  __set_errno (ENOSYS);
  return -1;
}
