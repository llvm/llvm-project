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
#include <sys/time.h>
#include <hurd.h>

/* Adjust the current time of day by the amount in DELTA.
   If OLDDELTA is not NULL, it is filled in with the amount
   of time adjustment remaining to be done from the last `__adjtime' call.
   This call is restricted to the super-user.  */
int
__adjtime (const struct timeval *delta, struct timeval *olddelta)
{
  error_t err;
  mach_port_t hostpriv;
  struct timeval dummy;

  err = __get_privileged_ports (&hostpriv, NULL);
  if (err)
    return __hurd_fail (EPERM);

  if (olddelta == NULL)
    olddelta = &dummy;

  err = __host_adjust_time (hostpriv,
			    /* `time_value_t' and `struct timeval' are in
                               fact identical with the names changed.  */
			    *(time_value_t *) delta,
			    (time_value_t *) olddelta);
  __mach_port_deallocate (__mach_task_self (), hostpriv);

  if (err)
    return __hurd_fail (err);
  return 0;
}

weak_alias (__adjtime, adjtime)
