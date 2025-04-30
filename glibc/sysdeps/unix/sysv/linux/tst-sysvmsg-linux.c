/* Basic tests for Linux SYSV message queue extensions.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <sys/ipc.h>
#include <sys/msg.h>
#include <errno.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

#include <support/check.h>
#include <support/temp_file.h>

#define MSGQ_MODE 0644

/* These are for the temporary file we generate.  */
static char *name;
static int msqid;

static void
remove_msq (void)
{
  /* Enforce message queue removal in case of early test failure.
     Ignore error since the msg may already have being removed.  */
  msgctl (msqid, IPC_RMID, NULL);
}

static void
do_prepare (int argc, char *argv[])
{
  TEST_VERIFY_EXIT (create_temp_file ("tst-sysvmsg.", &name) != -1);
}

#define PREPARE do_prepare

struct test_msginfo
{
  int msgmax;
  int msgmnb;
  int msgmni;
};

/* It tries to obtain some system-wide SysV messsage queue information from
   /proc to check against IPC_INFO/MSG_INFO.  The /proc only returns the
   tunables value of MSGMAX, MSGMNB, and MSGMNI.

   The kernel also returns constant value for MSGSSZ, MSGSEG and also MSGMAP,
   MSGPOOL, and MSGTQL (for IPC_INFO).  The issue to check them is they might
   change over kernel releases.  */

static int
read_proc_file (const char *file)
{
  FILE *f = fopen (file, "r");
  if (f == NULL)
    FAIL_UNSUPPORTED ("/proc is not mounted or %s is not available", file);

  int v;
  int r = fscanf (f, "%d", & v);
  TEST_VERIFY_EXIT (r == 1);

  fclose (f);
  return v;
}


/* Check if the message queue with IDX (index into the kernel's internal
   array) matches the one with KEY.  The CMD is either MSG_STAT or
   MSG_STAT_ANY.  */

static bool
check_msginfo (int idx, key_t key, int cmd)
{
  struct msqid_ds msginfo;
  int mid = msgctl (idx, cmd, &msginfo);
  /* Ignore unused array slot returned by the kernel or information from
     unknown message queue.  */
  if ((mid == -1 && errno == EINVAL) || mid != msqid)
    return false;

  if (mid == -1)
    FAIL_EXIT1 ("msgctl with %s failed: %m",
		cmd == MSG_STAT ? "MSG_STAT" : "MSG_STAT_ANY");

  TEST_COMPARE (msginfo.msg_perm.__key, key);
  TEST_COMPARE (msginfo.msg_perm.mode, MSGQ_MODE);
  TEST_COMPARE (msginfo.msg_qnum, 0);

  return true;
}

static int
do_test (void)
{
  atexit (remove_msq);

  key_t key = ftok (name, 'G');
  if (key == -1)
    FAIL_EXIT1 ("ftok failed: %m");

  msqid = msgget (key, MSGQ_MODE | IPC_CREAT);
  if (msqid == -1)
    FAIL_EXIT1 ("msgget failed: %m");

  struct test_msginfo tipcinfo;
  tipcinfo.msgmax = read_proc_file ("/proc/sys/kernel/msgmax");
  tipcinfo.msgmnb = read_proc_file ("/proc/sys/kernel/msgmnb");
  tipcinfo.msgmni = read_proc_file ("/proc/sys/kernel/msgmni");

  int msqidx;

  {
    struct msginfo ipcinfo;
    msqidx = msgctl (msqid, IPC_INFO, (struct msqid_ds *) &ipcinfo);
    if (msqidx == -1)
      FAIL_EXIT1 ("msgctl with IPC_INFO failed: %m");

    TEST_COMPARE (ipcinfo.msgmax, tipcinfo.msgmax);
    TEST_COMPARE (ipcinfo.msgmnb, tipcinfo.msgmnb);
    TEST_COMPARE (ipcinfo.msgmni, tipcinfo.msgmni);
  }

  /* Same as before but with MSG_INFO.  */
  {
    struct msginfo ipcinfo;
    msqidx = msgctl (msqid, MSG_INFO, (struct msqid_ds *) &ipcinfo);
    if (msqidx == -1)
      FAIL_EXIT1 ("msgctl with IPC_INFO failed: %m");

    TEST_COMPARE (ipcinfo.msgmax, tipcinfo.msgmax);
    TEST_COMPARE (ipcinfo.msgmnb, tipcinfo.msgmnb);
    TEST_COMPARE (ipcinfo.msgmni, tipcinfo.msgmni);
  }

  /* We check if the created message queue shows in global list.  */
  bool found = false;
  for (int i = 0; i <= msqidx; i++)
    {
      /* We can't tell apart if MSG_STAT_ANY is not supported (kernel older
	 than 4.17) or if the index used is invalid.  So it just check if the
	 value returned from a valid call matches the created message
	 queue.  */
      check_msginfo (i, key, MSG_STAT_ANY);

      if (check_msginfo (i, key, MSG_STAT))
	{
	  found = true;
	  break;
	}
    }

  if (!found)
    FAIL_EXIT1 ("msgctl with MSG_STAT/MSG_STAT_ANY could not find the "
		"created message queue");

  if (msgctl (msqid, IPC_RMID, NULL) == -1)
    FAIL_EXIT1 ("msgctl failed");

  return 0;
}

#include <support/test-driver.c>
