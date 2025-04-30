/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*
 * Four routines to be able to lock access to a file in the Linux
 * approved manner.  This seems to be the only method that works
 * over NFS:
 *
 *  __pg_make_lock_file( char* dir )
 *	Creates a uniquely-named file in the given directory.
 *	To work, this must be created on the same filesystem
 *	as the file which we are attempting to lock.
 *	If there are multiple processes running each competing for the
 *	same lock, each gets a unique file here.
 *	return -1 if it can't get a unique file in 50 tries
 *  __pg_get_lock( char* lname )
 *	The argument is the name of the lock.
 *	Each process tries to create a hard link with this name
 *	to its own uniquely-named file from __pg_make_lock_file().
 *	The one that succeeds is the new lock owner.  The others
 *	fail and try again.  There is a fail-over to handle the case
 *	where the process with the lock dies, which is inherently unsafe,
 *	but we haven't come up with a better solution.
 *  __pg_release_lock( char* lname )
 *	The argument is the same name for the lock.
 *	The lock is released by deleting (calling unlink) for the
 *	hard link we had just created.
 *  __pg_delete_lock_file()
 *	Clean up by deleting the uniquely named file we had created earlier.
 */

#ifndef _WIN64
  #include <unistd.h>
#else
  #include <Winsock2.h>
  #include <process.h>
  #define pid_t int
#endif
#include <fcntl.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include "lockfile.h"

/*
 * The name of the uniquely-named file and directory in which it is created
 */
static char *uname = NULL;
static char *udir = NULL;

/*
 * Some counts for debugging
 */
static long uwaiting;

#ifdef _WIN64
#define pid_t int
#endif
int
__pg_make_lock_file(char *dir)
{
  pid_t pid;
  char hostname[999];
  int r;
  FILE *F;
  udir = (char *)malloc(strlen(dir) + 1);
  strcpy(udir, dir);
  pid = getpid();
  r = gethostname(hostname, 998);
  if (r != 0) {
    fprintf(stderr, "gethostname fails\n");
    exit(1);
  }
  uname = (char *)malloc(strlen(hostname) + strlen(udir) + 25);
  r = 0;
  do {
    ++r;
    /* create a unique file */
    sprintf(uname, "%s/lock.%s.%ld%d", udir, hostname, (long)pid, r);
    F = fopen(uname, "w");
  } while (F == NULL && r < 51);
  if (F == NULL) {
    return -1;
  }
  fprintf(F, "%s\n", uname);
  fclose(F);
  uwaiting = 0;
  return 0;
} /* __pg_make_lock_file */

void
__pg_get_lock(char *lname)
{
  int r;
  struct stat ss, ss1;
  int count;
  char *fullname;
  if (udir == NULL || uname == NULL)
    return;
  fullname = (char *)malloc(strlen(lname) + strlen(udir) +
                            2); /* +1 for / +1 for null terminator */
  strcpy(fullname, udir);
  strcat(fullname, "/");
  strcat(fullname, lname);
  memset(&ss1, 0, sizeof(ss1));
  count = 0;
  do {
    r = link(uname, fullname);
    if (r != 0) {
      /* link had some problem, see if the number of links
       * to the new file is now two */
      r = stat(uname, &ss);
      if (r == 0) {
        if (ss.st_nlink != 2) {
          if (ss.st_nlink != 1) {
            fprintf(stderr, "get_lock: %d links to %s\n", (int)ss.st_nlink,
                    uname);
            exit(1);
          }
          /* see if we should forcefully TAKE the lock */
          r = lstat(fullname, &ss);
          if (count == 0) {
            memcpy(&ss1, &ss, sizeof(ss));
            count = 1;
          } else if (ss1.st_dev != ss.st_dev || ss1.st_ino != ss.st_ino ||
                     ss1.st_mode != ss.st_mode || ss1.st_uid != ss.st_uid ||
                     ss1.st_gid != ss.st_gid || ss1.st_size != ss.st_size ||
                     ss1.st_mtime != ss.st_mtime) {
            /* some else got there first */
            memcpy(&ss1, &ss, sizeof(ss));
            count = 1;
          } else {
            ++count;
            if (count > 20) {
              /* we've waited long enough. */
              r = unlink(fullname);
              /* ignore errors on the unlink */
            }
          }
          sleep(1);
          r = 1;
        }
      }
      if (r)
        ++uwaiting;
    }
  } while (r != 0);
  free(fullname);
} /* __pg_get_lock */

void
__pg_release_lock(char *lname)
{
  int r;
  char *fullname;
  if (udir == NULL || uname == NULL)
    return;
  fullname = (char *)malloc(strlen(lname) + strlen(udir) + 2);
  strcpy(fullname, udir);
  strcat(fullname, "/");
  strcat(fullname, lname);
  /* release the lock by deleting the link */
  r = unlink(fullname);
  if (r != 0) {
    fprintf(stderr, "release_lock: unlink %s fails\n", lname);
    exit(1);
  }
  free(fullname);
} /* __pg_release_lock */

void
__pg_delete_lock_file()
{
  int r;
  /* delete the unique file */
  r = unlink(uname);
  if (r != 0) {
    fprintf(stderr, "delete_lock_file: unlink %s fails\n", uname);
    exit(1);
  }
  free(uname);
  free(udir);
  uname = NULL;
  udir = NULL;
} /* __pg_delete_lock_file */
