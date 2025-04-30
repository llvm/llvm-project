/* Basic tests for Linux SYSV shared memory extensions.
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
#include <sys/shm.h>
#include <errno.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>
#include <inttypes.h>
#include <limits.h>

#include <support/check.h>
#include <support/temp_file.h>

#define SHM_MODE 0644

/* These are for the temporary file we generate.  */
static char *name;
static int shmid;
static long int pgsz;

static void
remove_shm (void)
{
  /* Enforce message queue removal in case of early test failure.
     Ignore error since the shm may already have being removed.  */
  shmctl (shmid, IPC_RMID, NULL);
}

static void
do_prepare (int argc, char *argv[])
{
  TEST_VERIFY_EXIT (create_temp_file ("tst-sysvshm.", &name) != -1);
}

#define PREPARE do_prepare

struct test_shminfo
{
  __syscall_ulong_t shmall;
  __syscall_ulong_t shmmax;
  __syscall_ulong_t shmmni;
};

/* It tries to obtain some system-wide SysV shared memory information from
   /proc to check against IPC_INFO/SHM_INFO.  The /proc only returns the
   tunables value of SHMALL, SHMMAX, and SHMMNI.  */

static uint64_t
read_proc_file (const char *file)
{
  FILE *f = fopen (file, "r");
  if (f == NULL)
    FAIL_UNSUPPORTED ("/proc is not mounted or %s is not available", file);

  /* Handle 32-bit binaries running on 64-bit kernels.  */
  uint64_t v;
  int r = fscanf (f, "%" SCNu64, &v);
  TEST_VERIFY_EXIT (r == 1);

  fclose (f);
  return v;
}


/* Check if the message queue with IDX (index into the kernel's internal
   array) matches the one with KEY.  The CMD is either SHM_STAT or
   SHM_STAT_ANY.  */

static bool
check_shminfo (int idx, key_t key, int cmd)
{
  struct shmid_ds shminfo;
  int sid = shmctl (idx, cmd, &shminfo);
  /* Ignore unused array slot returned by the kernel or information from
     unknown message queue.  */
  if ((sid == -1 && errno == EINVAL) || sid != shmid)
    return false;

  if (sid == -1)
    FAIL_EXIT1 ("shmctl with %s failed: %m",
		cmd == SHM_STAT ? "SHM_STAT" : "SHM_STAT_ANY");

  TEST_COMPARE (shminfo.shm_perm.__key, key);
  TEST_COMPARE (shminfo.shm_perm.mode, SHM_MODE);
  TEST_COMPARE (shminfo.shm_segsz, pgsz);

  return true;
}

static int
do_test (void)
{
  atexit (remove_shm);

  pgsz = sysconf (_SC_PAGESIZE);
  if (pgsz == -1)
    FAIL_EXIT1 ("sysconf (_SC_PAGESIZE) failed: %m");

  key_t key = ftok (name, 'G');
  if (key == -1)
    FAIL_EXIT1 ("ftok failed: %m");

  shmid = shmget (key, pgsz, IPC_CREAT | IPC_EXCL | SHM_MODE);
  if (shmid == -1)
    FAIL_EXIT1 ("shmget failed: %m");

  /* It does not check shmmax because kernel clamp its value to INT_MAX for:

     1. Compat symbols with IPC_64, i.e, 32-bit binaries running on 64-bit
        kernels.

     2. Default symbol without IPC_64 (defined as IPC_OLD within Linux) and
        glibc always use IPC_64 for 32-bit ABIs (to support 64-bit time_t).
        It means that 32-bit binaries running on 32-bit kernels will not see
        shmmax being clamped.

     And finding out whether the compat symbol is used would require checking
     the underlying kernel against the current ABI.  The shmall and shmmni
     already provided enough coverage.  */

  struct test_shminfo tipcinfo;
  tipcinfo.shmall = read_proc_file ("/proc/sys/kernel/shmall");
  tipcinfo.shmmni = read_proc_file ("/proc/sys/kernel/shmmni");

  int shmidx;

  /* Note: SHM_INFO does not return a shminfo, but rather a 'struct shm_info'.
     It is tricky to verify its values since the syscall returns system wide
     resources consumed by shared memory.  The shmctl implementation handles
     SHM_INFO as IPC_INFO, so the IPC_INFO test should validate SHM_INFO as
     well.  */

  {
    struct shminfo ipcinfo;
    shmidx = shmctl (shmid, IPC_INFO, (struct shmid_ds *) &ipcinfo);
    if (shmidx == -1)
      FAIL_EXIT1 ("shmctl with IPC_INFO failed: %m");

    TEST_COMPARE (ipcinfo.shmall, tipcinfo.shmall);
    TEST_COMPARE (ipcinfo.shmmni, tipcinfo.shmmni);
  }

  /* We check if the created shared memory shows in the global list.  */
  bool found = false;
  for (int i = 0; i <= shmidx; i++)
    {
      /* We can't tell apart if SHM_STAT_ANY is not supported (kernel older
	 than 4.17) or if the index used is invalid.  So it just check if
	 value returned from a valid call matches the created message
	 queue.  */
      check_shminfo (i, key, SHM_STAT_ANY);

      if (check_shminfo (i, key, SHM_STAT))
	{
	  found = true;
	  break;
	}
    }

  if (!found)
    FAIL_EXIT1 ("shmctl with SHM_STAT/SHM_STAT_ANY could not find the "
		"created shared memory");

  if (shmctl (shmid, IPC_RMID, NULL) == -1)
    FAIL_EXIT1 ("shmctl failed");

  return 0;
}

#include <support/test-driver.c>
