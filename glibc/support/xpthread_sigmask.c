/* pthread_sigmask with error checking.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <support/xsignal.h>
#include <support/support.h>

#include <unistd.h>

void
xpthread_sigmask (int how, const sigset_t *set, sigset_t *oldset)
{
  if (pthread_sigmask (how, set, oldset) != 0)
    {
      write_message ("error: pthread_setmask failed\n");
      /* Do not use exit because pthread_sigmask can be called from a
         signal handler.  */
      _exit (1);
    }
}
