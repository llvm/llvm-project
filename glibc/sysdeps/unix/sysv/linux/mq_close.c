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
#include <sysdep.h>
#include <shlib-compat.h>

/* Removes the association between message queue descriptor MQDES and its
   message queue.  */
int
__mq_close (mqd_t mqdes)
{
  return INLINE_SYSCALL (close, 1, mqdes);
}
versioned_symbol (libc, __mq_close, mq_close, GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (librt, GLIBC_2_3_4, GLIBC_2_34)
compat_symbol (libc, __mq_close, mq_close, GLIBC_2_3_4);
#endif
