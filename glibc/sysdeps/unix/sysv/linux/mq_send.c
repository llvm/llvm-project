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
#include <shlib-compat.h>
#include <stddef.h>
#include <sysdep.h>

/* Add message pointed by MSG_PTR to message queue MQDES.  */
int
__mq_send (mqd_t mqdes, const char *msg_ptr, size_t msg_len,
	 unsigned int msg_prio)
{
  return __mq_timedsend (mqdes, msg_ptr, msg_len, msg_prio, NULL);
}
versioned_symbol (libc, __mq_send, mq_send, GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (librt, GLIBC_2_3_4, GLIBC_2_34)
compat_symbol (librt, __mq_send, mq_send, GLIBC_2_3_4);
#endif
