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

#include <signal.h>
#include <errno.h>
#include <pthread.h>

/* Raise the signal SIG.  */
int
raise (int sig)
{
  int ret = __pthread_kill (__pthread_self (), sig);
  if (ret != 0)
    {
      __set_errno (ret);
      ret = -1;
    }
  return ret;
}
libc_hidden_def (raise)
weak_alias (raise, gsignal)
