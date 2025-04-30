/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jaku@redhat.com>, 2004.

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

#include <pthread.h>
#include <pwd.h>
#include <grp.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>


static pthread_barrier_t b3, b4;
static uid_t prev_ruid, prev_euid, prev_suid, nobody_uid;
static gid_t prev_rgid, prev_egid, prev_sgid, nobody_gid;
enum ACTION { PREPARE, SET, CHECK_BEFORE, CHECK_AFTER };
#define TESTNO(arg) ((long int) (arg) & 0xff)
#define THREADNO(arg) ((long int) (arg) >> 8)


static void
check_prev_uid (int tno)
{
  uid_t ruid, euid, suid;
  if (getresuid (&ruid, &euid, &suid) < 0)
    {
      printf ("getresuid failed: %d %m\n", tno);
      exit (1);
    }

  if (ruid != prev_ruid || euid != prev_euid || suid != prev_suid)
    {
      printf ("uids before in %d (%d %d %d) != (%d %d %d)\n", tno,
	      ruid, euid, suid, prev_ruid, prev_euid, prev_suid);
      exit (1);
    }
}


static void
check_prev_gid (int tno)
{
  gid_t rgid, egid, sgid;
  if (getresgid (&rgid, &egid, &sgid) < 0)
    {
      printf ("getresgid failed: %d %m\n", tno);
      exit (1);
    }

  if (rgid != prev_rgid || egid != prev_egid || sgid != prev_sgid)
    {
      printf ("gids before in %d (%d %d %d) != (%d %d %d)\n", tno,
	      rgid, egid, sgid, prev_rgid, prev_egid, prev_sgid);
      exit (1);
    }
}


static void
test_setuid1 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    return;

  if (action != CHECK_AFTER)
    check_prev_uid (tno);

  if (action == SET && setuid (nobody_uid) < 0)
    {
       printf ("setuid failed: %m\n");
       exit (1);
    }

  if (action != CHECK_BEFORE)
    {
      uid_t ruid, euid, suid;
      if (getresuid (&ruid, &euid, &suid) < 0)
	{
	  printf ("getresuid failed: %d %m\n", tno);
	  exit (1);
	}

      if (ruid != nobody_uid || euid != nobody_uid || suid != nobody_uid)
	{
	  printf ("after setuid %d (%d %d %d) != (%d %d %d)\n", tno,
		  ruid, euid, suid, nobody_uid, nobody_uid, nobody_uid);
	  exit (1);
	}
    }
}


static void
test_setuid2 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    {
      if (setresuid (nobody_uid, nobody_uid, -1) < 0)
	{
	  printf ("setresuid failed: %m\n");
	  exit (1);
	}

      prev_ruid = nobody_uid;
      prev_euid = nobody_uid;
      return;
    }

  if (action != CHECK_AFTER)
    check_prev_uid (tno);

  if (action == SET && setuid (prev_suid) < 0)
    {
      printf ("setuid failed: %m\n");
      exit (1);
    }

  if (action != CHECK_BEFORE)
    {
      uid_t ruid, euid, suid;
      if (getresuid (&ruid, &euid, &suid) < 0)
	{
	  printf ("getresuid failed: %d %m\n", tno);
	  exit (1);
	}

      if (ruid != nobody_uid || euid != prev_suid || suid != prev_suid)
	{
	  printf ("after setuid %d (%d %d %d) != (%d %d %d)\n", tno,
		  ruid, euid, suid, nobody_uid, prev_suid, prev_suid);
	  exit (1);
	}
    }
}


static void
test_seteuid1 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    return;

  if (action != CHECK_AFTER)
    check_prev_uid (tno);

  if (action == SET && seteuid (nobody_uid) < 0)
    {
       printf ("seteuid failed: %m\n");
       exit (1);
    }

  if (action != CHECK_BEFORE)
    {
      uid_t ruid, euid, suid;
      if (getresuid (&ruid, &euid, &suid) < 0)
	{
	  printf ("getresuid failed: %d %m\n", tno);
	  exit (1);
	}

      if (ruid != prev_ruid || euid != nobody_uid || suid != prev_suid)
	{
	  printf ("after seteuid %d (%d %d %d) != (%d %d %d)\n", tno,
		  ruid, euid, suid, prev_ruid, nobody_uid, prev_suid);
	  exit (1);
	}
    }
}


static void
test_seteuid2 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    {
      if (setresuid (nobody_uid, nobody_uid, -1) < 0)
	{
	  printf ("setresuid failed: %m\n");
	  exit (1);
	}

      prev_ruid = nobody_uid;
      prev_euid = nobody_uid;
      nobody_uid = prev_suid;
      return;
    }

  test_seteuid1 (action, tno);
}


static void
test_setreuid1 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    return;

  if (action != CHECK_AFTER)
    check_prev_uid (tno);

  if (action == SET && setreuid (-1, nobody_uid) < 0)
    {
       printf ("setreuid failed: %m\n");
       exit (1);
    }

  if (action != CHECK_BEFORE)
    {
      uid_t ruid, euid, suid, esuid;
      if (getresuid (&ruid, &euid, &suid) < 0)
	{
	  printf ("getresuid failed: %d %m\n", tno);
	  exit (1);
	}

      if (prev_ruid != nobody_uid)
	esuid = nobody_uid;
      else
	esuid = prev_suid;

      if (ruid != prev_ruid || euid != nobody_uid || suid != esuid)
	{
	  printf ("after setreuid %d (%d %d %d) != (%d %d %d)\n", tno,
		  ruid, euid, suid, prev_ruid, nobody_uid, esuid);
	  exit (1);
	}
    }
}


static void
test_setreuid2 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    return;

  if (action != CHECK_AFTER)
    check_prev_uid (tno);

  if (action == SET && setreuid (nobody_uid, -1) < 0)
    {
       printf ("setreuid failed: %m\n");
       exit (1);
    }

  if (action != CHECK_BEFORE)
    {
      uid_t ruid, euid, suid;
      if (getresuid (&ruid, &euid, &suid) < 0)
	{
	  printf ("getresuid failed: %d %m\n", tno);
	  exit (1);
	}

      if (ruid != nobody_uid || euid != prev_euid || suid != prev_euid)
	{
	  printf ("after setreuid %d (%d %d %d) != (%d %d %d)\n", tno,
		  ruid, euid, suid, nobody_uid, prev_euid, prev_euid);
	  exit (1);
	}
    }
}


static void
test_setreuid3 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    return;

  if (action != CHECK_AFTER)
    check_prev_uid (tno);

  if (action == SET && setreuid (nobody_uid, nobody_uid) < 0)
    {
       printf ("setreuid failed: %m\n");
       exit (1);
    }

  if (action != CHECK_BEFORE)
    {
      uid_t ruid, euid, suid;
      if (getresuid (&ruid, &euid, &suid) < 0)
	{
	  printf ("getresuid failed: %d %m\n", tno);
	  exit (1);
	}

      if (ruid != nobody_uid || euid != nobody_uid || suid != nobody_uid)
	{
	  printf ("after setreuid %d (%d %d %d) != (%d %d %d)\n", tno,
		  ruid, euid, suid, nobody_uid, nobody_uid, nobody_uid);
	  exit (1);
	}
    }
}


static void
test_setreuid4 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    {
      if (setresuid (nobody_uid, nobody_uid, -1) < 0)
	{
	  printf ("setresuid failed: %m\n");
	  exit (1);
	}

      prev_ruid = nobody_uid;
      prev_euid = nobody_uid;
      nobody_uid = prev_suid;
      return;
    }

  test_setreuid1 (action, tno);
}


static void
test_setresuid1 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    return;

  if (action != CHECK_AFTER)
    check_prev_uid (tno);

  if (action == SET && setresuid (-1, nobody_uid, -1) < 0)
    {
       printf ("setresuid failed: %m\n");
       exit (1);
    }

  if (action != CHECK_BEFORE)
    {
      uid_t ruid, euid, suid;
      if (getresuid (&ruid, &euid, &suid) < 0)
	{
	  printf ("getresuid failed: %d %m\n", tno);
	  exit (1);
	}

      if (ruid != prev_ruid || euid != nobody_uid || suid != prev_suid)
	{
	  printf ("after setresuid %d (%d %d %d) != (%d %d %d)\n", tno,
		  ruid, euid, suid, prev_ruid, nobody_uid, prev_suid);
	  exit (1);
	}
    }
}


static void
test_setresuid2 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    return;

  if (action != CHECK_AFTER)
    check_prev_uid (tno);

  if (action == SET && setresuid (prev_euid, nobody_uid, nobody_uid) < 0)
    {
       printf ("setresuid failed: %m\n");
       exit (1);
    }

  if (action != CHECK_BEFORE)
    {
      uid_t ruid, euid, suid;
      if (getresuid (&ruid, &euid, &suid) < 0)
	{
	  printf ("getresuid failed: %d %m\n", tno);
	  exit (1);
	}

      if (ruid != prev_euid || euid != nobody_uid || suid != nobody_uid)
	{
	  printf ("after setresuid %d (%d %d %d) != (%d %d %d)\n", tno,
		  ruid, euid, suid, prev_euid, nobody_uid, nobody_uid);
	  exit (1);
	}
    }
}


static void
test_setresuid3 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    return;

  if (action != CHECK_AFTER)
    check_prev_uid (tno);

  if (action == SET && setresuid (nobody_uid, nobody_uid, nobody_uid) < 0)
    {
       printf ("setresuid failed: %m\n");
       exit (1);
    }

  if (action != CHECK_BEFORE)
    {
      uid_t ruid, euid, suid;
      if (getresuid (&ruid, &euid, &suid) < 0)
	{
	  printf ("getresuid failed: %d %m\n", tno);
	  exit (1);
	}

      if (ruid != nobody_uid || euid != nobody_uid || suid != nobody_uid)
	{
	  printf ("after setresuid %d (%d %d %d) != (%d %d %d)\n", tno,
		  ruid, euid, suid, nobody_uid, nobody_uid, nobody_uid);
	  exit (1);
	}
    }
}


static void
test_setresuid4 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    {
      if (setresuid (nobody_uid, nobody_uid, -1) < 0)
	{
	  printf ("setresuid failed: %m\n");
	  exit (1);
	}

      prev_ruid = nobody_uid;
      prev_euid = nobody_uid;
      nobody_uid = prev_suid;
      return;
    }

  test_setresuid1 (action, tno);
}


static void
test_setgid1 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    return;

  if (action != CHECK_AFTER)
    check_prev_gid (tno);

  if (action == SET && setgid (nobody_gid) < 0)
    {
       printf ("setgid failed: %m\n");
       exit (1);
    }

  if (action != CHECK_BEFORE)
    {
      gid_t rgid, egid, sgid;
      if (getresgid (&rgid, &egid, &sgid) < 0)
	{
	  printf ("getresgid failed: %d %m\n", tno);
	  exit (1);
	}

      if (rgid != nobody_gid || egid != nobody_gid || sgid != nobody_gid)
	{
	  printf ("after setgid %d (%d %d %d) != (%d %d %d)\n", tno,
		  rgid, egid, sgid, nobody_gid, nobody_gid, nobody_gid);
	  exit (1);
	}
    }
}


static void
test_setgid2 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    {
      if (setresgid (nobody_gid, nobody_gid, -1) < 0)
	{
	  printf ("setresgid failed: %m\n");
	  exit (1);
	}

      prev_rgid = nobody_gid;
      prev_egid = nobody_gid;

      if (setresuid (nobody_uid, nobody_uid, -1) < 0)
	{
	  printf ("setresuid failed: %m\n");
	  exit (1);
	}

      prev_ruid = nobody_uid;
      prev_euid = nobody_uid;
      return;
    }

  if (action != CHECK_AFTER)
    check_prev_gid (tno);

  if (action == SET && setgid (prev_sgid) < 0)
    {
      printf ("setgid failed: %m\n");
      exit (1);
    }

  if (action != CHECK_BEFORE)
    {
      gid_t rgid, egid, sgid;
      if (getresgid (&rgid, &egid, &sgid) < 0)
	{
	  printf ("getresgid failed: %d %m\n", tno);
	  exit (1);
	}

      if (rgid != nobody_gid || egid != prev_sgid || sgid != prev_sgid)
	{
	  printf ("after setgid %d (%d %d %d) != (%d %d %d)\n", tno,
		  rgid, egid, sgid, nobody_gid, prev_sgid, prev_sgid);
	  exit (1);
	}
    }
}


static void
test_setegid1 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    return;

  if (action != CHECK_AFTER)
    check_prev_gid (tno);

  if (action == SET && setegid (nobody_gid) < 0)
    {
       printf ("setegid failed: %m\n");
       exit (1);
    }

  if (action != CHECK_BEFORE)
    {
      gid_t rgid, egid, sgid;
      if (getresgid (&rgid, &egid, &sgid) < 0)
	{
	  printf ("getresgid failed: %d %m\n", tno);
	  exit (1);
	}

      if (rgid != prev_rgid || egid != nobody_gid || sgid != prev_sgid)
	{
	  printf ("after setegid %d (%d %d %d) != (%d %d %d)\n", tno,
		  rgid, egid, sgid, prev_rgid, nobody_gid, prev_sgid);
	  exit (1);
	}
    }
}


static void
test_setegid2 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    {
      if (setresgid (nobody_gid, nobody_gid, -1) < 0)
	{
	  printf ("setresgid failed: %m\n");
	  exit (1);
	}

      prev_rgid = nobody_gid;
      prev_egid = nobody_gid;
      nobody_gid = prev_sgid;
      return;
    }

  test_setegid1 (action, tno);
}


static void
test_setregid1 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    return;

  if (action != CHECK_AFTER)
    check_prev_gid (tno);

  if (action == SET && setregid (-1, nobody_gid) < 0)
    {
       printf ("setregid failed: %m\n");
       exit (1);
    }

  if (action != CHECK_BEFORE)
    {
      gid_t rgid, egid, sgid, esgid;
      if (getresgid (&rgid, &egid, &sgid) < 0)
	{
	  printf ("getresgid failed: %d %m\n", tno);
	  exit (1);
	}

      if (prev_rgid != nobody_gid)
	esgid = nobody_gid;
      else
	esgid = prev_sgid;

      if (rgid != prev_rgid || egid != nobody_gid || sgid != esgid)
	{
	  printf ("after setregid %d (%d %d %d) != (%d %d %d)\n", tno,
		  rgid, egid, sgid, prev_rgid, nobody_gid, esgid);
	  exit (1);
	}
    }
}


static void
test_setregid2 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    return;

  if (action != CHECK_AFTER)
    check_prev_gid (tno);

  if (action == SET && setregid (nobody_gid, -1) < 0)
    {
       printf ("setregid failed: %m\n");
       exit (1);
    }

  if (action != CHECK_BEFORE)
    {
      gid_t rgid, egid, sgid;
      if (getresgid (&rgid, &egid, &sgid) < 0)
	{
	  printf ("getresgid failed: %d %m\n", tno);
	  exit (1);
	}

      if (rgid != nobody_gid || egid != prev_egid || sgid != prev_egid)
	{
	  printf ("after setregid %d (%d %d %d) != (%d %d %d)\n", tno,
		  rgid, egid, sgid, nobody_gid, prev_egid, prev_egid);
	  exit (1);
	}
    }
}


static void
test_setregid3 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    return;

  if (action != CHECK_AFTER)
    check_prev_gid (tno);

  if (action == SET && setregid (nobody_gid, nobody_gid) < 0)
    {
       printf ("setregid failed: %m\n");
       exit (1);
    }

  if (action != CHECK_BEFORE)
    {
      gid_t rgid, egid, sgid;
      if (getresgid (&rgid, &egid, &sgid) < 0)
	{
	  printf ("getresgid failed: %d %m\n", tno);
	  exit (1);
	}

      if (rgid != nobody_gid || egid != nobody_gid || sgid != nobody_gid)
	{
	  printf ("after setregid %d (%d %d %d) != (%d %d %d)\n", tno,
		  rgid, egid, sgid, nobody_gid, nobody_gid, nobody_gid);
	  exit (1);
	}
    }
}


static void
test_setregid4 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    {
      if (setresgid (nobody_gid, nobody_gid, -1) < 0)
	{
	  printf ("setresgid failed: %m\n");
	  exit (1);
	}

      prev_rgid = nobody_gid;
      prev_egid = nobody_gid;
      nobody_gid = prev_sgid;
      return;
    }

  test_setregid1 (action, tno);
}


static void
test_setresgid1 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    return;

  if (action != CHECK_AFTER)
    check_prev_gid (tno);

  if (action == SET && setresgid (-1, nobody_gid, -1) < 0)
    {
       printf ("setresgid failed: %m\n");
       exit (1);
    }

  if (action != CHECK_BEFORE)
    {
      gid_t rgid, egid, sgid;
      if (getresgid (&rgid, &egid, &sgid) < 0)
	{
	  printf ("getresgid failed: %d %m\n", tno);
	  exit (1);
	}

      if (rgid != prev_rgid || egid != nobody_gid || sgid != prev_sgid)
	{
	  printf ("after setresgid %d (%d %d %d) != (%d %d %d)\n", tno,
		  rgid, egid, sgid, prev_rgid, nobody_gid, prev_sgid);
	  exit (1);
	}
    }
}


static void
test_setresgid2 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    return;

  if (action != CHECK_AFTER)
    check_prev_gid (tno);

  if (action == SET && setresgid (prev_egid, nobody_gid, nobody_gid) < 0)
    {
       printf ("setresgid failed: %m\n");
       exit (1);
    }

  if (action != CHECK_BEFORE)
    {
      gid_t rgid, egid, sgid;
      if (getresgid (&rgid, &egid, &sgid) < 0)
	{
	  printf ("getresgid failed: %d %m\n", tno);
	  exit (1);
	}

      if (rgid != prev_egid || egid != nobody_gid || sgid != nobody_gid)
	{
	  printf ("after setresgid %d (%d %d %d) != (%d %d %d)\n", tno,
		  rgid, egid, sgid, prev_egid, nobody_gid, nobody_gid);
	  exit (1);
	}
    }
}


static void
test_setresgid3 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    return;

  if (action != CHECK_AFTER)
    check_prev_gid (tno);

  if (action == SET && setresgid (nobody_gid, nobody_gid, nobody_gid) < 0)
    {
       printf ("setresgid failed: %m\n");
       exit (1);
    }

  if (action != CHECK_BEFORE)
    {
      gid_t rgid, egid, sgid;
      if (getresgid (&rgid, &egid, &sgid) < 0)
	{
	  printf ("getresgid failed: %d %m\n", tno);
	  exit (1);
	}

      if (rgid != nobody_gid || egid != nobody_gid || sgid != nobody_gid)
	{
	  printf ("after setresgid %d (%d %d %d) != (%d %d %d)\n", tno,
		  rgid, egid, sgid, nobody_gid, nobody_gid, nobody_gid);
	  exit (1);
	}
    }
}


static void
test_setresgid4 (enum ACTION action, int tno)
{
  if (action == PREPARE)
    {
      if (setresgid (nobody_gid, nobody_gid, -1) < 0)
	{
	  printf ("setresgid failed: %m\n");
	  exit (1);
	}

      prev_rgid = nobody_gid;
      prev_egid = nobody_gid;
      nobody_gid = prev_sgid;
      return;
    }

  test_setresgid1 (action, tno);
}


static struct setuid_test
{
  const char *name;
  void (*test) (enum ACTION, int tno);
} setuid_tests[] =
{
  { "setuid1", test_setuid1 },
  { "setuid2", test_setuid2 },
  { "seteuid1", test_seteuid1 },
  { "seteuid2", test_seteuid2 },
  { "setreuid1", test_setreuid1 },
  { "setreuid2", test_setreuid2 },
  { "setreuid3", test_setreuid3 },
  { "setreuid4", test_setreuid4 },
  { "setresuid1", test_setresuid1 },
  { "setresuid2", test_setresuid2 },
  { "setresuid3", test_setresuid3 },
  { "setresuid4", test_setresuid4 },
  { "setgid1", test_setgid1 },
  { "setgid2", test_setgid2 },
  { "setegid1", test_setegid1 },
  { "setegid2", test_setegid2 },
  { "setregid1", test_setregid1 },
  { "setregid2", test_setregid2 },
  { "setregid3", test_setregid3 },
  { "setregid4", test_setregid4 },
  { "setresgid1", test_setresgid1 },
  { "setresgid2", test_setresgid2 },
  { "setresgid3", test_setresgid3 },
  { "setresgid4", test_setresgid4 }
};


static void *
tf2 (void *arg)
{
  int e = pthread_barrier_wait (&b4);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      exit (1);
    }

  setuid_tests[TESTNO (arg)].test (CHECK_AFTER, THREADNO (arg));
  return NULL;
}


static void *
tf (void *arg)
{
  setuid_tests[TESTNO (arg)].test (CHECK_BEFORE, THREADNO (arg));

  int e = pthread_barrier_wait (&b3);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      exit (1);
    }

  return tf2 (arg);
}


static int
do_one_test (long int testno)
{
  printf ("%s test\n", setuid_tests[testno].name);

  pid_t pid = fork ();
  if (pid == 0)
    {
      setuid_tests[testno].test (PREPARE, 0);
      setuid_tests[testno].test (SET, 0);
      exit (0);
    }

  if (pid < 0)
    {
      printf ("fork failed: %m\n");
      exit (1);
    }

  int status;
  if (waitpid (pid, &status, 0) < 0)
    {
      printf ("waitpid failed: %m\n");
      exit (1);
    }

  if (!WIFEXITED (status))
    {
      puts ("child did not exit");
      exit (1);
    }

  if (WEXITSTATUS (status))
    {
      printf ("skipping %s test\n", setuid_tests[testno].name);
      return 0;
    }

  pid = fork ();
  if (pid == 0)
    {
      setuid_tests[testno].test (PREPARE, 0);

      pthread_t th;
      int e = pthread_create (&th, NULL, tf, (void *) (testno | 0x100L));
      if (e != 0)
	{
	  printf ("create failed: %m\n");
	  exit (1);
	}

      pthread_t th2;
      e = pthread_create (&th2, NULL, tf, (void *) (testno | 0x200L));
      if (e != 0)
	{
	  printf ("create failed: %m\n");
	  exit (1);
	}

      e = pthread_barrier_wait (&b3);
      if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  puts ("barrier_wait failed");
	  exit (1);
	}

      setuid_tests[testno].test (SET, 0);

      pthread_t th3;
      e = pthread_create (&th3, NULL, tf2, (void *) (testno | 0x300L));
      if (e != 0)
	{
	  printf ("create failed: %m\n");
	  exit (1);
	}

      e = pthread_barrier_wait (&b4);
      if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  puts ("barrier_wait failed");
	  exit (1);
	}

      exit (0);
    }

  if (pid < 0)
    {
      printf ("fork failed: %m\n");
      exit (1);
    }

  if (waitpid (pid, &status, 0) < 0)
    {
      printf ("waitpid failed: %m\n");
      exit (1);
    }

  if (!WIFEXITED (status))
    {
      puts ("second child did not exit");
      exit (1);
    }

  if (WEXITSTATUS (status))
    exit (WEXITSTATUS (status));

  return 0;
}


static int
do_test (void)
{
  struct passwd *pwd = getpwnam ("nobody");
  if (pwd == NULL)
    {
      puts ("User nobody doesn't exist");
      return 0;
    }
  nobody_uid = pwd->pw_uid;
  nobody_gid = pwd->pw_gid;

  if (getresuid (&prev_ruid, &prev_euid, &prev_suid) < 0)
    {
      printf ("getresuid failed: %m\n");
      exit (1);
    }

  if (getresgid (&prev_rgid, &prev_egid, &prev_sgid) < 0)
    {
      printf ("getresgid failed: %m\n");
      exit (1);
    }

  if (prev_ruid == nobody_uid || prev_euid == nobody_uid
      || prev_suid == nobody_uid)
    {
      puts ("already running as user nobody, skipping tests");
      exit (0);
    }

  if (prev_rgid == nobody_gid || prev_egid == nobody_gid
      || prev_sgid == nobody_gid)
    {
      puts ("already running as group nobody, skipping tests");
      exit (0);
    }

  if (pthread_barrier_init (&b3, NULL, 3) != 0)
    {
      puts ("barrier_init failed");
      exit (1);
    }

  if (pthread_barrier_init (&b4, NULL, 4) != 0)
    {
      puts ("barrier_init failed");
      exit (1);
    }

  for (unsigned long int testno = 0;
       testno < sizeof (setuid_tests) / sizeof (setuid_tests[0]);
       ++testno)
    do_one_test (testno);
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
