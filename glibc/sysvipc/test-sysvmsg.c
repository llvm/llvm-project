/* Basic tests for SYSV message queue functions.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>

#include <test-sysvipc.h>

#include <support/support.h>
#include <support/check.h>
#include <support/temp_file.h>

#define TEXTSIZE 32
struct msgbuf_t
{
#ifdef _GNU_SOURCE
  __syscall_slong_t type;
#else
  long type;
#endif
  char buf[TEXTSIZE];
};

#define MSGTYPE 0x01020304
#define MSGDATA "0123456789"

/* These are for the temporary file we generate.  */
static char *name;
static int msqid;

static void
remove_msq (void)
{
  /* Enforce message queue removal in case of early test failure.
     Ignore error since the msgq may already have being removed.  */
  msgctl (msqid, IPC_RMID, NULL);
}

static void
do_prepare (int argc, char *argv[])
{
  int fd = create_temp_file ("tst-sysvmsg.", &name);
  if (fd == -1)
    FAIL_EXIT1 ("cannot create temporary file (errno=%d)", errno);
}

#define PREPARE do_prepare

/* It is not an extensive test, but rather a functional one aimed to check
   correct parameter passing on kernel.  */

#define MSGQ_MODE 0644

static int
do_test (void)
{
  atexit (remove_msq);

  key_t key = ftok (name, 'G');
  if (key == -1)
    FAIL_EXIT1 ("ftok failed");

  msqid = msgget (key, MSGQ_MODE | IPC_CREAT);
  if (msqid == -1)
    {
      if (errno == ENOSYS)
	FAIL_UNSUPPORTED ("msgget not supported");
      FAIL_EXIT1 ("msgget failed (errno=%d)", errno);
    }

  TEST_COMPARE (msgctl (msqid, first_msg_invalid_cmd (), NULL), -1);
  TEST_COMPARE (errno, EINVAL);

  /* Get message queue kernel information and do some sanity checks.  */
  struct msqid_ds msginfo;
  if (msgctl (msqid, IPC_STAT, &msginfo) == -1)
    FAIL_EXIT1 ("msgctl with IPC_STAT failed (errno=%d)", errno);

  if (msginfo.msg_perm.__key != key)
    FAIL_EXIT1 ("msgid_ds::msg_perm::key (%d) != %d",
		(int) msginfo.msg_perm.__key, (int) key);
  if (msginfo.msg_perm.mode != MSGQ_MODE)
    FAIL_EXIT1 ("msgid_ds::msg_perm::mode (%o) != %o",
		msginfo.msg_perm.mode, MSGQ_MODE);
  if (msginfo.msg_qnum != 0)
    FAIL_EXIT1 ("msgid_ds::msg_qnum (%lu) != 0",
		(long unsigned) msginfo.msg_qnum);

  /* Check if last argument (IPC_NOWAIT) is correctly handled.  */
  struct msgbuf_t msg2rcv = { 0 };
  if (msgrcv (msqid, &msg2rcv, sizeof (msg2rcv.buf), MSGTYPE,
	      IPC_NOWAIT) == -1
      && errno != ENOMSG)
    FAIL_EXIT1 ("msgrcv failed (errno=%d)", errno);

  struct msgbuf_t msg2snd = { MSGTYPE, MSGDATA };
  if (msgsnd (msqid, &msg2snd, sizeof (msg2snd.buf), 0) == -1)
    FAIL_EXIT1 ("msgsnd failed (errno=%d)", errno);

  if (msgrcv (msqid, &msg2rcv, sizeof (msg2rcv.buf), MSGTYPE, 0) == -1)
    FAIL_EXIT1 ("msgrcv failed (errno=%d)", errno);

  int ret = 0;
  if (strncmp (msg2snd.buf, msg2rcv.buf, TEXTSIZE) != 0)
    ret = 1;

  if (msgctl (msqid, IPC_RMID, NULL) == -1)
    FAIL_EXIT1 ("msgctl failed");

  return ret;
}

#include <support/test-driver.c>
