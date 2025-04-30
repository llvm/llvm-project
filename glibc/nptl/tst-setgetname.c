/* Test pthread_setname_np and pthread_getname_np.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

/* New name of process.  */
#define NEW_NAME "setname"

/* Name of process which is one byte too big
   e.g. 17 bytes including null-terminator  */
#define BIG_NAME       "....V....X....XV"

/* Longest name of a process
   e.g. 16 bytes including null-terminator.  */
#define LONGEST_NAME   "....V....X....X"

/* One less than longest name with unique
   characters to detect modification.  */
#define CANARY_NAME    "abcdefghijklmn"

/* On Linux the maximum length of the name of a task *including* the null
   terminator.  */
#define TASK_COMM_LEN 16

/* On Linux we can read this task's name from /proc.  */
int
get_self_comm (long tid, char *buf, size_t len)
{
  int res = 0;
#define FMT "/proc/self/task/%lu/comm"
  char fname[sizeof (FMT) + 32];
  sprintf (fname, FMT, (unsigned long) tid);

  int fd = open (fname, O_RDONLY);
  if (fd == -1)
    return errno;

  ssize_t n = read (fd, (void *) buf, len);
  if (n < 0)
    res = errno;
  else
    {
      if (buf[n - 1] == '\n')
        buf[n - 1] = '\0';
      else if (n == len)
        res = ERANGE;
      else
        buf[n] = '\0';
    }

  close (fd);
  return res;
}

int
do_test (int argc, char **argv)
{
  pthread_t self;
  int res;
  int ret = 0;
  char name[TASK_COMM_LEN];
  char name_check[TASK_COMM_LEN];

  memset (name, '\0', TASK_COMM_LEN);
  memset (name_check, '\0', TASK_COMM_LEN);

  /* Test 1: Get the name of the task via pthread_getname_np and /proc
     and verify that they both match.  */
  self = pthread_self ();
  res = pthread_getname_np (self, name, TASK_COMM_LEN);

  if (res == 0)
    {
      res = get_self_comm (gettid (), name_check, TASK_COMM_LEN);

      if (res == 0)
       {
         if (strncmp (name, name_check, strlen (BIG_NAME)) == 0)
           printf ("PASS: Test 1 - pthread_getname_np and /proc agree.\n");
         else
           {
             printf ("FAIL: Test 1 - pthread_getname_np and /proc differ"
                     " i.e. %s != %s\n", name, name_check);
             ret++;
           }
       }
      else
       {
         printf ("FAIL: Test 1 - unable read task name via proc.\n");
         ret++;
        }
    }
  else
    {
      printf ("FAIL: Test 1 - pthread_getname_np failed with error %d\n", res);
      ret++;
    }

  /* Test 2: Test setting the name and then independently verify it
             was set.  */
  res = pthread_setname_np (self, NEW_NAME);

  if (res == 0)
    {
      res = get_self_comm (gettid (), name_check, TASK_COMM_LEN);
      if (res == 0)
        {
         if (strncmp (NEW_NAME, name_check, strlen (BIG_NAME)) == 0)
           printf ("PASS: Test 2 - Value used in pthread_setname_np and"
                   " /proc agree.\n");
         else
           {
             printf ("FAIL: Test 2 - Value used in pthread_setname_np"
		     " and /proc differ i.e. %s != %s\n",
		     NEW_NAME, name_check);
             ret++;
           }
        }
      else
       {
         printf ("FAIL: Test 2 - unable to read task name via proc.\n");
         ret++;
        }
    }
  else
    {
      printf ("FAIL: Test 2 - pthread_setname_np failed with error %d\n", res);
      ret++;
    }

  /* Test 3: Test setting a name that is one-byte too big.  */
  res = pthread_getname_np (self, name, TASK_COMM_LEN);

  if (res == 0)
    {
      res = pthread_setname_np (self, BIG_NAME);
      if (res != 0)
        {
         if (res == ERANGE)
           {
             printf ("PASS: Test 3 - pthread_setname_np returned ERANGE"
                     " for a process name that was too long.\n");

             /* Verify the old name didn't change.  */
             res = get_self_comm (gettid (), name_check, TASK_COMM_LEN);
             if (res == 0)
               {
                 if (strncmp (name, name_check, strlen (BIG_NAME)) == 0)
                   printf ("PASS: Test 3 - Original name unchanged after"
                           " pthread_setname_np returned ERANGE.\n");
                 else
                   {
                     printf ("FAIL: Test 3 - Original name changed after"
                             " pthread_setname_np returned ERANGE"
                             " i.e. %s != %s\n",
                             name, name_check);
                     ret++;
                   }
               }
             else
               {
                 printf ("FAIL: Test 3 - unable to read task name.\n");
                 ret++;
               }
           }
         else
           {
             printf ("FAIL: Test 3 - Wrong error returned"
		     " i.e. ERANGE != %d\n", res);
             ret++;
           }
        }
      else
        {
         printf ("FAIL: Test 3 - Too-long name accepted by"
	         " pthread_setname_np.\n");
         ret++;
        }
    }
  else
    {
      printf ("FAIL: Test 3 - Unable to get original name.\n");
      ret++;
    }

  /* Test 4: Verify that setting the longest name works.  */
  res = pthread_setname_np (self, LONGEST_NAME);

  if (res == 0)
    {
      res = get_self_comm (gettid (), name_check, TASK_COMM_LEN);
      if (res == 0)
        {
         if (strncmp (LONGEST_NAME, name_check, strlen (BIG_NAME)) == 0)
           printf ("PASS: Test 4 - Longest name set via pthread_setname_np"
                   " agrees with /proc.\n");
         else
           {
             printf ("FAIL: Test 4 - Value used in pthread_setname_np and /proc"
		     " differ i.e. %s != %s\n", LONGEST_NAME, name_check);
             ret++;
           }
        }
      else
       {
         printf ("FAIL: Test 4 - unable to read task name via proc.\n");
         ret++;
        }
    }
  else
    {
      printf ("FAIL: Test 4 - pthread_setname_np failed with error %d\n", res);
      ret++;
    }

  /* Test 5: Verify that getting a long name into a small buffer fails.  */
  strncpy (name, CANARY_NAME, strlen (CANARY_NAME) + 1);

  /* Claim the buffer length is strlen (LONGEST_NAME).  This is one character
     too small to hold LONGEST_NAME *and* the null terminator.  We should get
     back ERANGE and name should be unmodified.  */
  res = pthread_getname_np (self, name, strlen (LONGEST_NAME));

  if (res != 0)
    {
      if (res == ERANGE)
        {
	  if (strncmp (CANARY_NAME, name, strlen (BIG_NAME)) == 0)
	    {
	      printf ("PASS: Test 5 - ERANGE and buffer unmodified.\n");
	    }
	  else
	    {
	      printf ("FAIL: Test 5 - Original buffer modified.\n");
	      ret++;
	    }
        }
      else
        {
	  printf ("FAIL: Test 5 - Did not return ERANGE for small buffer.\n");
	  ret++;
        }
    }
  else
    {
      printf ("FAIL: Test 5 - Returned name longer than buffer.\n");
      ret++;
    }

  /* Test 6: Lastly make sure we can read back the longest name.  */
  res = pthread_getname_np (self, name, strlen (LONGEST_NAME) + 1);

  if (res == 0)
    {
      if (strncmp (LONGEST_NAME, name, strlen (BIG_NAME)) == 0)
        {
	  printf ("PASS: Test 6 - Read back longest name correctly.\n");
        }
      else
        {
	  printf ("FAIL: Test 6 - Read \"%s\" instead of longest name.\n",
		  name);
	  ret++;
        }
    }
  else
    {
      printf ("FAIL: Test 6 - pthread_getname_np failed with error %d\n", res);
      ret++;
    }

  return ret;
}

#include <test-skeleton.c>
