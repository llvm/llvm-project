/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#ifndef _SYS_MSG_H
# error "Never use <bits/msq.h> directly; include <sys/msg.h> instead."
#endif

#include <bits/types.h>
#include <bits/ipctypes.h>

/* Define options for message queue functions.  */
#define MSG_NOERROR	010000	/* no error if message is too big */
#ifdef __USE_GNU
# define MSG_EXCEPT	020000	/* recv any msg except of specified type */
#endif

/* Types used in the structure definition.  */
typedef unsigned short int msgqnum_t;
typedef unsigned short int msglen_t;


/* Structure of record for one message inside the kernel.
   The type `struct msg' is opaque.  */
struct msqid_ds
{
  struct ipc_perm msg_perm;	/* structure describing operation permission */
  struct msg *__msg_first;	/* pointer to first message on queue */
  struct msg *__msg_last;	/* pointer to last message on queue */
  __time_t msg_stime;		/* time of last msgsnd command */
  __time_t msg_rtime;		/* time of last msgrcv command */
  __time_t msg_ctime;		/* time of last change */
  struct __wait_queue *__wwait;	/* ??? */
  struct __wait_queue *__rwait;	/* ??? */
  unsigned short int __msg_cbytes;/* current number of bytes on queue */
  msgqnum_t msg_qnum;		/* number of messages currently on queue */
  msglen_t msg_qbytes;		/* max number of bytes allowed on queue */
  __ipc_pid_t msg_lspid;	/* pid of last msgsnd() */
  __ipc_pid_t msg_lrpid;	/* pid of last msgrcv() */
};

#ifdef __USE_MISC

# define msg_cbytes	__msg_cbytes

/* ipcs ctl commands */
# define MSG_STAT 11
# define MSG_INFO 12

/* buffer for msgctl calls IPC_INFO, MSG_INFO */
struct msginfo
  {
    int msgpool;
    int msgmap;
    int msgmax;
    int msgmnb;
    int msgmni;
    int msgssz;
    int msgtql;
    unsigned short int msgseg;
  };

#endif /* __USE_MISC */
