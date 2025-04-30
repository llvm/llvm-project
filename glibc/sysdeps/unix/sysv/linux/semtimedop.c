/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, August 1995.

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

#include <sys/sem.h>
#include <ipc_priv.h>
#include <sysdep.h>
#include <errno.h>

static int
semtimedop_syscall (int semid, struct sembuf *sops, size_t nsops,
		    const struct __timespec64 *timeout)
{
#ifdef __NR_semtimedop_time64
  return INLINE_SYSCALL_CALL (semtimedop_time64, semid, sops, nsops, timeout);
#elif defined __ASSUME_DIRECT_SYSVIPC_SYSCALLS && defined __NR_semtimedop
  return INLINE_SYSCALL_CALL (semtimedop, semid, sops, nsops, timeout);
#else
  return INLINE_SYSCALL_CALL (ipc, IPCOP_semtimedop, semid,
			      SEMTIMEDOP_IPC_ARGS (nsops, sops, timeout));
#endif
}

/* Perform user-defined atomical operation of array of semaphores.  */
int
__semtimedop64 (int semid, struct sembuf *sops, size_t nsops,
		const struct __timespec64 *timeout)
{
#ifdef __ASSUME_TIME64_SYSCALLS
  return semtimedop_syscall (semid, sops, nsops, timeout);
#else
  bool need_time64 = timeout != NULL && !in_time_t_range (timeout->tv_sec);
  if (need_time64)
    {
      int r = semtimedop_syscall (semid, sops, nsops, timeout);
      if (r == 0 || errno != ENOSYS)
	return r;
      __set_errno (EOVERFLOW);
      return -1;
    }

  struct timespec ts32, *pts32 = NULL;
  if (timeout != NULL)
    {
      ts32 = valid_timespec64_to_timespec (*timeout);
      pts32 = &ts32;
    }
# ifdef __ASSUME_DIRECT_SYSVIPC_SYSCALLS
  return INLINE_SYSCALL_CALL (semtimedop, semid, sops, nsops, pts32);
# else
  return INLINE_SYSCALL_CALL (ipc, IPCOP_semtimedop, semid,
			      SEMTIMEDOP_IPC_ARGS (nsops, sops, pts32));
# endif
#endif
}
#if __TIMESIZE != 64
libc_hidden_def (__semtimedop64)

int
__semtimedop (int semid, struct sembuf *sops, size_t nsops,
	      const struct timespec *timeout)
{
  struct __timespec64 ts64, *pts64 = NULL;
  if (timeout != NULL)
    {
      ts64 = valid_timespec_to_timespec64 (*timeout);
      pts64 = &ts64;
    }
  return __semtimedop64 (semid, sops, nsops, pts64);
}
#endif
weak_alias (__semtimedop, semtimedop)
