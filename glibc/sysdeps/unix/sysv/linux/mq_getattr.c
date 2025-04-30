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
#include <stddef.h>
#include <sysdep.h>
#include <shlib-compat.h>

/* Query status and attributes of message queue MQDES.  */
int
__mq_getattr (mqd_t mqdes, struct mq_attr *mqstat)
{
  return mq_setattr (mqdes, NULL, mqstat);
}
versioned_symbol (libc, __mq_getattr, mq_getattr, GLIBC_2_34);
libc_hidden_ver (__mq_getattr, mq_getattr)
#if OTHER_SHLIB_COMPAT (librt, GLIBC_2_3_4, GLIBC_2_34)
compat_symbol (librt, __mq_getattr, mq_getattr, GLIBC_2_3_4);
#endif
