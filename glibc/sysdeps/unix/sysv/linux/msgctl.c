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

#include <sys/msg.h>
#include <ipc_priv.h>
#include <sysdep.h>
#include <shlib-compat.h>
#include <errno.h>
#include <linux/posix_types.h>  /* For __kernel_mode_t.  */

/* POSIX states ipc_perm mode should have type of mode_t.  */
_Static_assert (sizeof ((struct msqid_ds){0}.msg_perm.mode)
		== sizeof (mode_t),
		"sizeof (msqid_ds.msg_perm.mode) != sizeof (mode_t)");

#if __IPC_TIME64 == 0
typedef struct msqid_ds msgctl_arg_t;
#else
# include <struct_kernel_msqid64_ds.h>

static void
msqid64_to_kmsqid64 (const struct __msqid64_ds *msqid64,
		     struct kernel_msqid64_ds *kmsqid)
{
  kmsqid->msg_perm       = msqid64->msg_perm;
  kmsqid->msg_stime      = msqid64->msg_stime;
  kmsqid->msg_stime_high = msqid64->msg_stime >> 32;
  kmsqid->msg_rtime      = msqid64->msg_rtime;
  kmsqid->msg_rtime_high = msqid64->msg_rtime >> 32;
  kmsqid->msg_ctime      = msqid64->msg_ctime;
  kmsqid->msg_ctime_high = msqid64->msg_ctime >> 32;
  kmsqid->msg_cbytes     = msqid64->msg_cbytes;
  kmsqid->msg_qnum       = msqid64->msg_qnum;
  kmsqid->msg_qbytes     = msqid64->msg_qbytes;
  kmsqid->msg_lspid      = msqid64->msg_lspid;
  kmsqid->msg_lrpid      = msqid64->msg_lrpid;
}

static void
kmsqid64_to_msqid64 (const struct kernel_msqid64_ds *kmsqid,
		     struct __msqid64_ds *msqid64)
{
  msqid64->msg_perm   = kmsqid->msg_perm;
  msqid64->msg_stime  = kmsqid->msg_stime
		        | ((__time64_t) kmsqid->msg_stime_high << 32);
  msqid64->msg_rtime  = kmsqid->msg_rtime
		        | ((__time64_t) kmsqid->msg_rtime_high << 32);
  msqid64->msg_ctime  = kmsqid->msg_ctime
		        | ((__time64_t) kmsqid->msg_ctime_high << 32);
  msqid64->msg_cbytes = kmsqid->msg_cbytes;
  msqid64->msg_qnum   = kmsqid->msg_qnum;
  msqid64->msg_qbytes = kmsqid->msg_qbytes;
  msqid64->msg_lspid  = kmsqid->msg_lspid;
  msqid64->msg_lrpid  = kmsqid->msg_lrpid;
}

typedef struct kernel_msqid64_ds msgctl_arg_t;
#endif

static int
msgctl_syscall (int msqid, int cmd, msgctl_arg_t *buf)
{
#ifdef __ASSUME_DIRECT_SYSVIPC_SYSCALLS
  return INLINE_SYSCALL_CALL (msgctl, msqid, cmd | __IPC_64, buf);
#else
  return INLINE_SYSCALL_CALL (ipc, IPCOP_msgctl, msqid, cmd | __IPC_64, 0,
			      buf);
#endif
}

int
__msgctl64 (int msqid, int cmd, struct __msqid64_ds *buf)
{
#if __IPC_TIME64
  struct kernel_msqid64_ds ksemid, *arg = NULL;
#else
  msgctl_arg_t *arg;
#endif

  switch (cmd)
    {
    case IPC_RMID:
      arg = NULL;
      break;

    case IPC_SET:
    case IPC_STAT:
    case MSG_STAT:
    case MSG_STAT_ANY:
#if __IPC_TIME64
      if (buf != NULL)
	{
	  msqid64_to_kmsqid64 (buf, &ksemid);
	  arg = &ksemid;
	}
# ifdef __ASSUME_SYSVIPC_BROKEN_MODE_T
      if (cmd == IPC_SET)
	arg->msg_perm.mode *= 0x10000U;
# endif
#else
      arg = buf;
#endif
      break;

    case IPC_INFO:
    case MSG_INFO:
      /* This is a Linux extension where kernel returns a 'struct msginfo'
	 instead.  */
      arg = (__typeof__ (arg)) buf;
      break;

    default:
      __set_errno (EINVAL);
      return -1;
    }

  int ret = msgctl_syscall (msqid, cmd, arg);
  if (ret < 0)
    return ret;

  switch (cmd)
    {
    case IPC_STAT:
    case MSG_STAT:
    case MSG_STAT_ANY:
#ifdef __ASSUME_SYSVIPC_BROKEN_MODE_T
      arg->msg_perm.mode >>= 16;
#else
      /* Old Linux kernel versions might not clear the mode padding.  */
      if (sizeof ((struct msqid_ds){0}.msg_perm.mode)
          != sizeof (__kernel_mode_t))
	arg->msg_perm.mode &= 0xFFFF;
#endif

#if __IPC_TIME64
      kmsqid64_to_msqid64 (arg, buf);
#endif
    }

  return ret;
}
#if __TIMESIZE != 64
libc_hidden_def (__msgctl64)

static void
msqid_to_msqid64 (struct __msqid64_ds *mq64, const struct msqid_ds *mq)
{
  mq64->msg_perm   = mq->msg_perm;
  mq64->msg_stime  = mq->msg_stime
		     | ((__time64_t) mq->__msg_stime_high << 32);
  mq64->msg_rtime  = mq->msg_rtime
		     | ((__time64_t) mq->__msg_rtime_high << 32);
  mq64->msg_ctime  = mq->msg_ctime
		     | ((__time64_t) mq->__msg_ctime_high << 32);
  mq64->msg_cbytes = mq->msg_cbytes;
  mq64->msg_qnum   = mq->msg_qnum;
  mq64->msg_qbytes = mq->msg_qbytes;
  mq64->msg_lspid  = mq->msg_lspid;
  mq64->msg_lrpid  = mq->msg_lrpid;
}

static void
msqid64_to_msqid (struct msqid_ds *mq, const struct __msqid64_ds *mq64)
{
  mq->msg_perm         = mq64->msg_perm;
  mq->msg_stime        = mq64->msg_stime;
  mq->__msg_stime_high = 0;
  mq->msg_rtime        = mq64->msg_rtime;
  mq->__msg_rtime_high = 0;
  mq->msg_ctime        = mq64->msg_ctime;
  mq->__msg_ctime_high = 0;
  mq->msg_cbytes       = mq64->msg_cbytes;
  mq->msg_qnum         = mq64->msg_qnum;
  mq->msg_qbytes       = mq64->msg_qbytes;
  mq->msg_lspid        = mq64->msg_lspid;
  mq->msg_lrpid        = mq64->msg_lrpid;
}

int
__msgctl (int msqid, int cmd, struct msqid_ds *buf)
{
  struct __msqid64_ds msqid64, *buf64 = NULL;
  if (buf != NULL)
    {
      /* This is a Linux extension where kernel returns a 'struct msginfo'
	 instead.  */
      if (cmd == IPC_INFO || cmd == MSG_INFO)
	buf64 = (struct __msqid64_ds *) buf;
      else
	{
	  msqid_to_msqid64 (&msqid64, buf);
	  buf64 = &msqid64;
	}
    }

  int ret = __msgctl64 (msqid, cmd, buf64);
  if (ret < 0)
    return ret;

  switch (cmd)
    {
    case IPC_STAT:
    case MSG_STAT:
    case MSG_STAT_ANY:
      msqid64_to_msqid (buf, buf64);
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
versioned_symbol (libc, __msgctl, msgctl, DEFAULT_VERSION);

#if defined __ASSUME_SYSVIPC_BROKEN_MODE_T \
    && SHLIB_COMPAT (libc, GLIBC_2_2, GLIBC_2_31)
int
attribute_compat_text_section
__msgctl_mode16 (int msqid, int cmd, struct msqid_ds *buf)
{
  return msgctl_syscall (msqid, cmd, (msgctl_arg_t *) buf);
}
compat_symbol (libc, __msgctl_mode16, msgctl, GLIBC_2_2);
#endif

#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_2)
struct __old_msqid_ds
{
  struct __old_ipc_perm msg_perm;	/* structure describing operation permission */
  struct msg *__msg_first;		/* pointer to first message on queue */
  struct msg *__msg_last;		/* pointer to last message on queue */
  __time_t msg_stime;			/* time of last msgsnd command */
  __time_t msg_rtime;			/* time of last msgrcv command */
  __time_t msg_ctime;			/* time of last change */
  struct wait_queue *__wwait;		/* ??? */
  struct wait_queue *__rwait;		/* ??? */
  unsigned short int __msg_cbytes;	/* current number of bytes on queue */
  unsigned short int msg_qnum;		/* number of messages currently on queue */
  unsigned short int msg_qbytes;	/* max number of bytes allowed on queue */
  __ipc_pid_t msg_lspid;		/* pid of last msgsnd() */
  __ipc_pid_t msg_lrpid;		/* pid of last msgrcv() */
};

int
attribute_compat_text_section
__old_msgctl (int msqid, int cmd, struct __old_msqid_ds *buf)
{
#if defined __ASSUME_DIRECT_SYSVIPC_SYSCALLS \
    && !defined __ASSUME_SYSVIPC_DEFAULT_IPC_64
  /* For architecture that have wire-up msgctl but also have __IPC_64 to a
     value different than default (0x0) it means the compat symbol used the
     __NR_ipc syscall.  */
  return INLINE_SYSCALL_CALL (msgctl, msqid, cmd, buf);
#else
  return INLINE_SYSCALL_CALL (ipc, IPCOP_msgctl, msqid, cmd, 0, buf);
#endif
}
compat_symbol (libc, __old_msgctl, msgctl, GLIBC_2_0);
#endif
