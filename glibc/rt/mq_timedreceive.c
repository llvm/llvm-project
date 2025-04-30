/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
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
#include <mqueue.h>

/* Receive the oldest from highest priority messages in message queue
   MQDES, stop waiting if ABS_TIMEOUT expires.  */
ssize_t
__mq_timedreceive (mqd_t mqdes, char *__restrict msg_ptr, size_t msg_len,
		 unsigned int *__restrict msg_prio,
		 const struct timespec *__restrict abs_timeout)
{
  __set_errno (ENOSYS);
  return -1;
}
hidden_def (__mq_timedreceive)
weak_alias (__mq_timedreceive, mq_timedreceive)
hidden_weak (mq_timedreceive)
stub_warning (mq_timedreceive)
