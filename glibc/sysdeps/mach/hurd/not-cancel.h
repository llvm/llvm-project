/* Uncancelable versions of cancelable interfaces.  Hurd version.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2003.

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

#ifndef NOT_CANCEL_H
# define NOT_CANCEL_H

#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>
#include <time.h>
#include <sys/uio.h>
#include <hurd.h>
#include <hurd/fd.h>

/* Non cancellable close syscall.  */
__typeof (__close) __close_nocancel;

void __close_nocancel_nostatus (int fd);

/* Non cancellable open syscall.  */
__typeof (__open) __open_nocancel;
/* open64 is just the same as open for us.  */
#define __open64_nocancel(...) \
  __open_nocancel (__VA_ARGS__)

/* Non cancellable openat syscall.  */
__typeof (__openat) __openat_nocancel;
/* open64 is just the same as open for us.  */
#define __openat64_nocancel(...) \
  __openat_nocancel (__VA_ARGS__)

/* Non cancellable read syscall.  */
__typeof (__read) __read_nocancel;

/* Non cancellable pread syscall (LFS version).  */
__typeof (__pread64) __pread64_nocancel;

/* Non cancellable write syscall.  */
__typeof (__write) __write_nocancel;

/* Non cancellable pwrite syscall (LFS version).  */
__typeof (__pwrite64) __pwrite64_nocancel;

/* Non cancellable writev syscall.  */
__typeof (__writev) __writev_nocancel;

/* Non cancellable writev syscall with no status.  */
void __writev_nocancel_nostatus (int fd, const struct iovec *vector, int count);

/* Non cancellable wait4 syscall.  */
__typeof (__wait4) __wait4_nocancel;

# define __waitpid_nocancel(pid, stat_loc, options) \
  __wait4_nocancel (pid, stat_loc, options, NULL)

/* Non cancellable fcntl syscall.  */
__typeof (__fcntl) __fcntl_nocancel;
/* fcntl64 is just the same as fcntl for us.  */
#define __fcntl64_nocancel(...) \
  __fcntl_nocancel (__VA_ARGS__)

#if IS_IN (libc)
hidden_proto (__close_nocancel)
hidden_proto (__close_nocancel_nostatus)
hidden_proto (__open_nocancel)
hidden_proto (__openat_nocancel)
hidden_proto (__read_nocancel)
hidden_proto (__pread64_nocancel)
hidden_proto (__write_nocancel)
hidden_proto (__pwrite64_nocancel)
hidden_proto (__writev_nocancel)
hidden_proto (__writev_nocancel_nostatus)
hidden_proto (__wait4_nocancel)
hidden_proto (__fcntl_nocancel)
#endif

#endif /* NOT_CANCEL_H  */
