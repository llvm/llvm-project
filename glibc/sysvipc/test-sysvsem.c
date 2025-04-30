/* Basic tests for SYSV semaphore functions.
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

#include <intprops.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>

#include <test-sysvipc.h>

#include <support/support.h>
#include <support/check.h>
#include <support/temp_file.h>
#include <support/xtime.h>
#include <support/xsignal.h>

/* These are for the temporary file we generate.  */
static char *name;
static int semid;

static void
remove_sem (void)
{
  /* Enforce message queue removal in case of early test failure.
     Ignore error since the sem may already have being removed.  */
  semctl (semid, 0, IPC_RMID, 0);
}

static void
do_prepare (int argc, char *argv[])
{
  int fd = create_temp_file ("tst-sysvsem.", &name);
  if (fd == -1)
    FAIL_EXIT1 ("cannot create temporary file (errno=%d)", errno);
}

#define PREPARE do_prepare

/* It is not an extensive test, but rather a functional one aimed to check
   correct parameter passing on kernel.  */

#define SEM_MODE 0644

union semun
{
  int val;
  struct semid_ds *buf;
  unsigned short  *array;
};

static int
do_test (void)
{
  atexit (remove_sem);

  key_t key = ftok (name, 'G');
  if (key == -1)
    FAIL_EXIT1 ("ftok failed");

  semid = semget(key, 1, IPC_CREAT | IPC_EXCL | SEM_MODE);
  if (semid == -1)
    {
      if (errno == ENOSYS)
	FAIL_UNSUPPORTED ("msgget not supported");
      FAIL_EXIT1 ("semget failed (errno=%d)", errno);
    }

  TEST_COMPARE (semctl (semid, 0, first_sem_invalid_cmd (), NULL), -1);
  TEST_COMPARE (errno, EINVAL);

  /* Get semaphore kernel information and do some sanity checks.  */
  struct semid_ds seminfo;
  if (semctl (semid, 0, IPC_STAT, (union semun) { .buf = &seminfo }) == -1)
    FAIL_EXIT1 ("semctl with IPC_STAT failed (errno=%d)", errno);

  if (seminfo.sem_perm.__key != key)
    FAIL_EXIT1 ("semid_ds::sem_perm::key (%d) != %d",
		(int) seminfo.sem_perm.__key, (int) key);
  if (seminfo.sem_perm.mode != SEM_MODE)
    FAIL_EXIT1 ("semid_ds::sem_perm::mode (%o) != %o",
		seminfo.sem_perm.mode, SEM_MODE);
  if (seminfo.sem_nsems != 1)
    FAIL_EXIT1 ("semid_ds::sem_nsems (%lu) != 1",
		(long unsigned) seminfo.sem_nsems);

  /* Some lock/unlock basic tests.  */
  struct sembuf sb1 = { 0, 1, 0 };
  if (semop (semid, &sb1, 1) == -1)
    FAIL_EXIT1 ("semop failed (errno=%i)", errno);

  struct sembuf sb2 = { 0, -1, 0 };
  if (semop (semid, &sb2, 1) == -1)
    FAIL_EXIT1 ("semop failed (errno=%i)", errno);

#ifdef _GNU_SOURCE
  /* Set a time for half a second.  The semaphore operation should timeout
     with EAGAIN.  */
  {
    struct timespec ts = { 0 /* sec */, 500000000 /* nsec */ };
    if (semtimedop (semid, &sb2, 1, &ts) != -1
        || (errno != EAGAIN && errno != ENOSYS))
      FAIL_EXIT1 ("semtimedop succeed or returned errno != {EAGAIN,ENOSYS} "
		  "(errno=%i)", errno);
  }

  {
    support_create_timer (0, 100000000, false, NULL);
    struct timespec ts = { TYPE_MAXIMUM (time_t), 0 };
    TEST_COMPARE (semtimedop (semid, &sb2, 1, &ts), -1);
    TEST_VERIFY (errno == EINTR || errno == EOVERFLOW);
  }
#endif

  /* Finally free up the semnaphore resource.  */
  if (semctl (semid, 0, IPC_RMID, 0) == -1)
    FAIL_EXIT1 ("semctl failed (errno=%d)", errno);

  return 0;
}

#include <support/test-driver.c>
