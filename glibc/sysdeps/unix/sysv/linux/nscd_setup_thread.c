/* Setup of nscd worker threads.  Linux verison.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2004.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published
   by the Free Software Foundation; version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

#include <string.h>
#include <unistd.h>
#include <nscd.h>
#include <sysdep.h>


int
setup_thread (struct database_dyn *db)
{
  /* Only supported when NPTL is used.  */
  char buf[100];
  if (confstr (_CS_GNU_LIBPTHREAD_VERSION, buf, sizeof (buf)) >= sizeof (buf)
      || strncmp (buf, "NPTL", 4) != 0)
    return 0;

  /* Do not try this at home, kids.  We play with the SETTID address
     even thought the process is multi-threaded.  This can only work
     since none of the threads ever terminates.  */
  int r = INTERNAL_SYSCALL_CALL (set_tid_address,
				 &db->head->nscd_certainly_running);
  if (!INTERNAL_SYSCALL_ERROR_P (r))
    /* We know the kernel can reset this field when nscd terminates.
       So, set the field to a nonzero value which indicates that nscd
       is certainly running and clients can skip the test.  */
    return db->head->nscd_certainly_running = 1;

  return 0;
}
