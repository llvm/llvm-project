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
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <sysdep.h>
#include <shlib-compat.h>

/* Establish connection between a process and a message queue NAME and
   return message queue descriptor or (mqd_t) -1 on error.  OFLAG determines
   the type of access used.  If O_CREAT is on OFLAG, the third argument is
   taken as a `mode_t', the mode of the created message queue, and the fourth
   argument is taken as `struct mq_attr *', pointer to message queue
   attributes.  If the fourth argument is NULL, default attributes are
   used.  */
mqd_t
__mq_open (const char *name, int oflag, ...)
{
  if (name[0] != '/')
    return INLINE_SYSCALL_ERROR_RETURN_VALUE (EINVAL);

  mode_t mode = 0;
  struct mq_attr *attr = NULL;
  if (oflag & O_CREAT)
    {
      va_list ap;

      va_start (ap, oflag);
      mode = va_arg (ap, mode_t);
      attr = va_arg (ap, struct mq_attr *);
      va_end (ap);
    }

  return INLINE_SYSCALL (mq_open, 4, name + 1, oflag, mode, attr);
}
versioned_symbol (libc, __mq_open, mq_open, GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (librt, GLIBC_2_3_4, GLIBC_2_34)
compat_symbol (libc, __mq_open, mq_open, GLIBC_2_3_4);
#endif

mqd_t
___mq_open_2 (const char *name, int oflag)
{
  if (oflag & O_CREAT)
    __fortify_fail ("invalid mq_open call: O_CREAT without mode and attr");

  return __mq_open (name, oflag);
}
versioned_symbol (libc, ___mq_open_2, __mq_open_2, GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (librt, GLIBC_2_7, GLIBC_2_34)
compat_symbol (libc, ___mq_open_2, __mq_open_2, GLIBC_2_7);
#endif
