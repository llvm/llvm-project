/* mq_setattr system call wrapper.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <mqueue.h>
#include <shlib-compat.h>
#include <sysdep.h>

int
__mq_setattr (mqd_t mqdes, const struct mq_attr *mqstat,
              struct mq_attr * omqstat)
{
  return INLINE_SYSCALL_CALL (mq_getsetattr, mqdes, mqstat, omqstat);
}
versioned_symbol (libc, __mq_setattr, mq_setattr, GLIBC_2_34);
libc_hidden_ver (__mq_setattr, mq_setattr)
#if OTHER_SHLIB_COMPAT (librt, GLIBC_2_3_4, GLIBC_2_34)
compat_symbol (librt, __mq_setattr, mq_setattr, GLIBC_2_3_4);
#endif
