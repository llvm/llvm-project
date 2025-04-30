/* Message-writing for the dynamic linker.  Linux version.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#include <sys/uio.h>
#include <sysdep.h>

/* This is used from only one place: dl-misc.c:_dl_debug_vdprintf.
   Hence it's in a header with the expectation it will be inlined.

   This is writev, but with a constraint added and others loosened:

   1. Under RTLD_PRIVATE_ERRNO, it must not clobber the private errno
      when another thread holds the dl_load_lock.
   2. It is not obliged to detect and report errors at all.
   3. It's not really obliged to deliver a single atomic write
      (though it may be preferable).  */

static inline void
_dl_writev (int fd, const struct iovec *iov, size_t niov)
{
  INTERNAL_SYSCALL_CALL (writev, fd, iov, niov);
}
