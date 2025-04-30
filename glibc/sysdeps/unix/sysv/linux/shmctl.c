/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.ai.mit.edu>, August 1995.

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

#include <sys/shm.h>
#include <stdarg.h>
#include <ipc_priv.h>
#include <sysdep.h>
#include <shlib-compat.h>
#include <errno.h>
#include <linux/posix_types.h>  /* For __kernel_mode_t.  */

/* POSIX states ipc_perm mode should have type of mode_t.  */
_Static_assert (sizeof ((struct shmid_ds){0}.shm_perm.mode)
		== sizeof (mode_t),
		"sizeof (shmid_ds.shm_perm.mode) != sizeof (mode_t)");

#if __IPC_TIME64 == 0
typedef struct shmid_ds shmctl_arg_t;
#else
# include <struct_kernel_shmid64_ds.h>

static void
shmid64_to_kshmid64 (const struct __shmid64_ds *shmid64,
		     struct kernel_shmid64_ds *kshmid)
{
  kshmid->shm_perm       = shmid64->shm_perm;
  kshmid->shm_segsz      = shmid64->shm_segsz;
  kshmid->shm_atime      = shmid64->shm_atime;
  kshmid->shm_atime_high = shmid64->shm_atime >> 32;
  kshmid->shm_dtime      = shmid64->shm_dtime;
  kshmid->shm_dtime_high = shmid64->shm_dtime >> 32;
  kshmid->shm_ctime      = shmid64->shm_ctime;
  kshmid->shm_ctime_high = shmid64->shm_ctime >> 32;
  kshmid->shm_cpid       = shmid64->shm_cpid;
  kshmid->shm_lpid       = shmid64->shm_lpid;
  kshmid->shm_nattch     = shmid64->shm_nattch;
}

static void
kshmid64_to_shmid64 (const struct kernel_shmid64_ds *kshmid,
		     struct __shmid64_ds *shmid64)
{
  shmid64->shm_perm   = kshmid->shm_perm;
  shmid64->shm_segsz  = kshmid->shm_segsz;
  shmid64->shm_atime  = kshmid->shm_atime
		        | ((__time64_t) kshmid->shm_atime_high << 32);
  shmid64->shm_dtime  = kshmid->shm_dtime
		        | ((__time64_t) kshmid->shm_dtime_high << 32);
  shmid64->shm_ctime  = kshmid->shm_ctime
		        | ((__time64_t) kshmid->shm_ctime_high << 32);
  shmid64->shm_cpid   = kshmid->shm_cpid;
  shmid64->shm_lpid   = kshmid->shm_lpid;
  shmid64->shm_nattch = kshmid->shm_nattch;
}

typedef struct kernel_shmid64_ds shmctl_arg_t;
#endif

static int
shmctl_syscall (int shmid, int cmd, shmctl_arg_t *buf)
{
#ifdef __ASSUME_DIRECT_SYSVIPC_SYSCALLS
  return INLINE_SYSCALL_CALL (shmctl, shmid, cmd | __IPC_64, buf);
#else
  return INLINE_SYSCALL_CALL (ipc, IPCOP_shmctl, shmid, cmd | __IPC_64, 0,
			      buf);
#endif
}

/* Provide operations to control over shared memory segments.  */
int
__shmctl64 (int shmid, int cmd, struct __shmid64_ds *buf)
{
#if __IPC_TIME64
  struct kernel_shmid64_ds kshmid, *arg = NULL;
#else
  shmctl_arg_t *arg;
#endif

  switch (cmd)
    {
    case IPC_RMID:
    case SHM_LOCK:
    case SHM_UNLOCK:
      arg = NULL;
      break;

    case IPC_SET:
    case IPC_STAT:
    case SHM_STAT:
    case SHM_STAT_ANY:
#if __IPC_TIME64
      if (buf != NULL)
	{
	  shmid64_to_kshmid64 (buf, &kshmid);
	  arg = &kshmid;
	}
# ifdef __ASSUME_SYSVIPC_BROKEN_MODE_T
      if (cmd == IPC_SET)
        arg->shm_perm.mode *= 0x10000U;
# endif
#else
      arg = buf;
#endif
      break;

    case IPC_INFO:
    case SHM_INFO:
      /* This is a Linux extension where kernel expects either a
	 'struct shminfo' (IPC_INFO) or 'struct shm_info' (SHM_INFO).  */
      arg = (__typeof__ (arg)) buf;
      break;

    default:
      __set_errno (EINVAL);
      return -1;
    }


  int ret = shmctl_syscall (shmid, cmd, arg);
  if (ret < 0)
    return ret;

  switch (cmd)
    {
      case IPC_STAT:
      case SHM_STAT:
      case SHM_STAT_ANY:
#ifdef __ASSUME_SYSVIPC_BROKEN_MODE_T
        arg->shm_perm.mode >>= 16;
#else
      /* Old Linux kernel versions might not clear the mode padding.  */
      if (sizeof ((struct shmid_ds){0}.shm_perm.mode)
	  != sizeof (__kernel_mode_t))
	arg->shm_perm.mode &= 0xFFFF;
#endif

#if __IPC_TIME64
      kshmid64_to_shmid64 (arg, buf);
#endif
    }

  return ret;
}
#if __TIMESIZE != 64
libc_hidden_def (__shmctl64)

static void
shmid_to_shmid64 (struct __shmid64_ds *shm64, const struct shmid_ds *shm)
{
  shm64->shm_perm   = shm->shm_perm;
  shm64->shm_segsz  = shm->shm_segsz;
  shm64->shm_atime  = shm->shm_atime
		      | ((__time64_t) shm->__shm_atime_high << 32);
  shm64->shm_dtime  = shm->shm_dtime
		      | ((__time64_t) shm->__shm_dtime_high << 32);
  shm64->shm_ctime  = shm->shm_ctime
		      | ((__time64_t) shm->__shm_ctime_high << 32);
  shm64->shm_cpid   = shm->shm_cpid;
  shm64->shm_lpid   = shm->shm_lpid;
  shm64->shm_nattch = shm->shm_nattch;
}

static void
shmid64_to_shmid (struct shmid_ds *shm, const struct __shmid64_ds *shm64)
{
  shm->shm_perm         = shm64->shm_perm;
  shm->shm_segsz        = shm64->shm_segsz;
  shm->shm_atime        = shm64->shm_atime;
  shm->__shm_atime_high = 0;
  shm->shm_dtime        = shm64->shm_dtime;
  shm->__shm_dtime_high = 0;
  shm->shm_ctime        = shm64->shm_ctime;
  shm->__shm_ctime_high = 0;
  shm->shm_cpid         = shm64->shm_cpid;
  shm->shm_lpid         = shm64->shm_lpid;
  shm->shm_nattch       = shm64->shm_nattch;
}

int
__shmctl (int shmid, int cmd, struct shmid_ds *buf)
{
  struct __shmid64_ds shmid64, *buf64 = NULL;
  if (buf != NULL)
    {
      /* This is a Linux extension where kernel expects either a
	 'struct shminfo' (IPC_INFO) or 'struct shm_info' (SHM_INFO).  */
      if (cmd == IPC_INFO || cmd == SHM_INFO)
	buf64 = (struct __shmid64_ds *) buf;
      else
	{
	  shmid_to_shmid64 (&shmid64, buf);
	  buf64 = &shmid64;
	}
    }

  int ret = __shmctl64 (shmid, cmd, buf64);
  if (ret < 0)
    return ret;

  switch (cmd)
    {
      case IPC_STAT:
      case SHM_STAT:
      case SHM_STAT_ANY:
	shmid64_to_shmid (buf, buf64);
    }

  return ret;
}
#endif

#ifndef DEFAULT_VERSION
# ifndef __ASSUME_SYSVIPC_BROKEN_MODE_T
#  define DEFAULT_VERSION GLIBC_2_2
# else
#  define DEFAULT_VERSION GLIBC_2_31
# endif
#endif

versioned_symbol (libc, __shmctl, shmctl, DEFAULT_VERSION);

#if defined __ASSUME_SYSVIPC_BROKEN_MODE_T \
    && SHLIB_COMPAT (libc, GLIBC_2_2, GLIBC_2_31)
int
attribute_compat_text_section
__shmctl_mode16 (int shmid, int cmd, struct shmid_ds *buf)
{
  return shmctl_syscall (shmid, cmd, (shmctl_arg_t *) buf);
}
compat_symbol (libc, __shmctl_mode16, shmctl, GLIBC_2_2);
#endif

#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_2)
struct __old_shmid_ds
{
  struct __old_ipc_perm shm_perm;	/* operation permission struct */
  int shm_segsz;			/* size of segment in bytes */
  __time_t shm_atime;			/* time of last shmat() */
  __time_t shm_dtime;			/* time of last shmdt() */
  __time_t shm_ctime;			/* time of last change by shmctl() */
  __ipc_pid_t shm_cpid;			/* pid of creator */
  __ipc_pid_t shm_lpid;			/* pid of last shmop */
  unsigned short int shm_nattch;	/* number of current attaches */
  unsigned short int __shm_npages;	/* size of segment (pages) */
  unsigned long int *__shm_pages;	/* array of ptrs to frames -> SHMMAX */
  struct vm_area_struct *__attaches;	/* descriptors for attaches */
};

int
attribute_compat_text_section
__old_shmctl (int shmid, int cmd, struct __old_shmid_ds *buf)
{
#if defined __ASSUME_DIRECT_SYSVIPC_SYSCALLS \
    && !defined __ASSUME_SYSVIPC_DEFAULT_IPC_64
  /* For architecture that have wire-up shmctl but also have __IPC_64 to a
     value different than default (0x0), it means the compat symbol used the
     __NR_ipc syscall.  */
  return INLINE_SYSCALL_CALL (shmctl, shmid, cmd, buf);
#else
  return INLINE_SYSCALL_CALL (ipc, IPCOP_shmctl, shmid, cmd, 0, buf);
#endif
}
compat_symbol (libc, __old_shmctl, shmctl, GLIBC_2_0);
#endif
