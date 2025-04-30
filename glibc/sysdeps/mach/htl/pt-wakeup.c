/* Wakeup a thread.  Mach version.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#include <assert.h>
#include <errno.h>

#include <mach.h>
#include <mach/message.h>

#include <pt-internal.h>

/* Wakeup THREAD.  */
void
__pthread_wakeup (struct __pthread *thread)
{
  error_t err;

  err = __mach_msg (&thread->wakeupmsg, MACH_SEND_MSG | MACH_SEND_TIMEOUT,
		    sizeof (thread->wakeupmsg), 0, MACH_PORT_NULL,
		    0, MACH_PORT_NULL);
  assert_perror (err);
}
