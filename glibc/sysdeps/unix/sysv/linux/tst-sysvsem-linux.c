/* Basic tests for Linux SYSV semaphore extensions.
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
#include <sys/sem.h>
#include <errno.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

#include <support/check.h>
#include <support/temp_file.h>

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
  TEST_VERIFY_EXIT (create_temp_file ("tst-sysvsem.", &name) != -1);
}

#define PREPARE do_prepare

#define SEM_MODE 0644

union semun
{
  int val;
  struct semid_ds *buf;
  unsigned short  *array;
  struct seminfo *__buf;
};

struct test_seminfo
{
  int semmsl;
  int semmns;
  int semopm;
  int semmni;
};

/* It tries to obtain some system-wide SysV semaphore information from /proc
   to check against IPC_INFO/SEM_INFO.  The /proc only returns the tunables
   value of SEMMSL, SEMMNS, SEMOPM, and SEMMNI.

   The kernel also returns constant value for SEMVMX, SEMMNU, SEMMAP, SEMUME,
   and also SEMUSZ and SEMAEM (for IPC_INFO).  The issue to check them is they
   might change over kernel releases.  */

static void
read_sem_stat (struct test_seminfo *tseminfo)
{
  FILE *f = fopen ("/proc/sys/kernel/sem", "r");
  if (f == NULL)
    FAIL_UNSUPPORTED ("/proc is not mounted or /proc/sys/kernel/sem is not "
		      "available");

  int r = fscanf (f, "%d %d %d %d",
		  &tseminfo->semmsl, &tseminfo->semmns, &tseminfo->semopm,
		  &tseminfo->semmni);
  TEST_VERIFY_EXIT (r == 4);

  fclose (f);
}


/* Check if the semaphore with IDX (index into the kernel's internal array)
   matches the one with KEY.  The CMD is either SEM_STAT or SEM_STAT_ANY.  */

static bool
check_seminfo (int idx, key_t key, int cmd)
{
  struct semid_ds seminfo;
  int sid = semctl (idx, 0, cmd, (union semun) { .buf = &seminfo });
  /* Ignore unused array slot returned by the kernel or information from
     unknown semaphores.  */
  if ((sid == -1 && errno == EINVAL) || sid != semid)
    return false;

  if (sid == -1)
    FAIL_EXIT1 ("semctl with SEM_STAT failed (errno=%d)", errno);

  TEST_COMPARE (seminfo.sem_perm.__key, key);
  TEST_COMPARE (seminfo.sem_perm.mode, SEM_MODE);
  TEST_COMPARE (seminfo.sem_nsems, 1);

  return true;
}

static int
do_test (void)
{
  atexit (remove_sem);

  key_t key = ftok (name, 'G');
  if (key == -1)
    FAIL_EXIT1 ("ftok failed: %m");

  semid = semget (key, 1, IPC_CREAT | IPC_EXCL | SEM_MODE);
  if (semid == -1)
    FAIL_EXIT1 ("semget failed: %m");

  struct test_seminfo tipcinfo;
  read_sem_stat (&tipcinfo);

  int semidx;

  {
    struct seminfo ipcinfo;
    semidx = semctl (semid, 0, IPC_INFO, (union semun) { .__buf = &ipcinfo });
    if (semidx == -1)
      FAIL_EXIT1 ("semctl with IPC_INFO failed: %m");

    TEST_COMPARE (ipcinfo.semmsl, tipcinfo.semmsl);
    TEST_COMPARE (ipcinfo.semmns, tipcinfo.semmns);
    TEST_COMPARE (ipcinfo.semopm, tipcinfo.semopm);
    TEST_COMPARE (ipcinfo.semmni, tipcinfo.semmni);
  }

  /* Same as before but with SEM_INFO.  */
  {
    struct seminfo ipcinfo;
    semidx = semctl (semid, 0, SEM_INFO, (union semun) { .__buf = &ipcinfo });
    if (semidx == -1)
      FAIL_EXIT1 ("semctl with IPC_INFO failed: %m");

    TEST_COMPARE (ipcinfo.semmsl, tipcinfo.semmsl);
    TEST_COMPARE (ipcinfo.semmns, tipcinfo.semmns);
    TEST_COMPARE (ipcinfo.semopm, tipcinfo.semopm);
    TEST_COMPARE (ipcinfo.semmni, tipcinfo.semmni);
  }

  /* We check if the created semaphore shows in the system-wide status.  */
  bool found = false;
  for (int i = 0; i <= semidx; i++)
    {
      /* We can't tell apart if SEM_STAT_ANY is not supported (kernel older
	 than 4.17) or if the index used is invalid.  So it just check if
	 value returned from a valid call matches the created semaphore.  */
      check_seminfo (i, key, SEM_STAT_ANY);

      if (check_seminfo (i, key, SEM_STAT))
	{
	  found = true;
	  break;
	}
    }

  if (!found)
    FAIL_EXIT1 ("semctl with SEM_STAT/SEM_STAT_ANY could not find the "
		"created  semaphore");

  if (semctl (semid, 0, IPC_RMID, 0) == -1)
    FAIL_EXIT1 ("semctl failed: %m");

  return 0;
}

#include <support/test-driver.c>
