/* Check if effective user id can access file
   Copyright (C) 1990-2021 Free Software Foundation, Inc.
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

/* Written by David MacKenzie and Torbjorn Granlund.
   Adapted for GNU C library by Roland McGrath.  */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <sys/types.h>
#include <sys/stat.h>

#ifdef S_IEXEC
# ifndef S_IXUSR
#  define S_IXUSR S_IEXEC
# endif
# ifndef S_IXGRP
#  define S_IXGRP (S_IEXEC >> 3)
# endif
# ifndef S_IXOTH
#  define S_IXOTH (S_IEXEC >> 6)
# endif
#endif /* S_IEXEC */

#if defined HAVE_UNISTD_H || defined _LIBC
# include <unistd.h>
#endif

#ifndef _POSIX_VERSION
uid_t getuid ();
gid_t getgid ();
uid_t geteuid ();
gid_t getegid ();
#endif /* not POSIX_VERSION */

#include <errno.h>
#ifndef errno
extern int errno;
#endif
#ifndef __set_errno
# define __set_errno(val) errno = (val)
#endif

#if defined EACCES && !defined EACCESS
# define EACCESS EACCES
#endif

#ifndef F_OK
# define F_OK 0
# define X_OK 1
# define W_OK 2
# define R_OK 4
#endif

#if !defined S_IROTH && defined R_OK
# define S_IROTH R_OK
#endif
#if !defined S_IWOTH && defined W_OK
# define S_IWOTH W_OK
#endif
#if !defined S_IXOTH && defined X_OK
# define S_IXOTH X_OK
#endif


#ifdef _LIBC

# define group_member __group_member
# define euidaccess __euidaccess

#else

/* The user's real user id. */
static uid_t uid;

/* The user's real group id. */
static gid_t gid;

/* The user's effective user id. */
static uid_t euid;

/* The user's effective group id. */
static gid_t egid;

/* Nonzero if UID, GID, EUID, and EGID have valid values. */
static int have_ids;

# ifdef HAVE_GETGROUPS
int group_member ();
# else
#  define group_member(gid)	0
# endif

#endif


/* Return 0 if the user has permission of type MODE on file PATH;
   otherwise, return -1 and set `errno' to EACCESS.
   Like access, except that it uses the effective user and group
   id's instead of the real ones, and it does not check for read-only
   filesystem, text busy, etc. */

int
euidaccess (const char *path, int mode)
{
  struct __stat64_t64 stats;
  int granted;

#ifdef	_LIBC
  uid_t euid;
  gid_t egid;
#else
  if (have_ids == 0)
    {
      have_ids = 1;
      uid = getuid ();
      gid = getgid ();
      euid = geteuid ();
      egid = getegid ();
    }

  if (uid == euid && gid == egid)
    /* If we are not set-uid or set-gid, access does the same.  */
    return access (path, mode);
#endif

  if (__stat64_time64 (path, &stats))
    return -1;

  mode &= (X_OK | W_OK | R_OK);	/* Clear any bogus bits. */
#if R_OK != S_IROTH || W_OK != S_IWOTH || X_OK != S_IXOTH
  ?error Oops, portability assumptions incorrect.
#endif

  if (mode == F_OK)
    return 0;			/* The file exists. */

#ifdef	_LIBC
  /* Now we need the IDs.  */
  euid = __geteuid ();
  egid = __getegid ();

  if (__getuid () == euid && __getgid () == egid)
    /* If we are not set-uid or set-gid, access does the same.  */
    return __access (path, mode);
#endif

  /* The super-user can read and write any file, and execute any file
     that anyone can execute. */
  if (euid == 0 && ((mode & X_OK) == 0
		    || (stats.st_mode & (S_IXUSR | S_IXGRP | S_IXOTH))))
    return 0;

  if (euid == stats.st_uid)
    granted = (unsigned int) (stats.st_mode & (mode << 6)) >> 6;
  else if (egid == stats.st_gid || group_member (stats.st_gid))
    granted = (unsigned int) (stats.st_mode & (mode << 3)) >> 3;
  else
    granted = (stats.st_mode & mode);
  /* XXX Add support for ACLs.  */
  if (granted == mode)
    return 0;
  __set_errno (EACCESS);
  return -1;
}
#undef euidaccess
#undef eaccess
#ifdef weak_alias
weak_alias (__euidaccess, euidaccess)
weak_alias (__euidaccess, eaccess)
#endif

#ifdef TEST
# include <stdio.h>
# include <errno.h>
# include "error.h"

char *program_name;

int
main (int argc, char **argv)
{
  char *file;
  int mode;
  int err;

  program_name = argv[0];
  if (argc < 3)
    abort ();
  file = argv[1];
  mode = atoi (argv[2]);

  err = euidaccess (file, mode);
  printf ("%d\n", err);
  if (err != 0)
    error (0, errno, "%s", file);
  exit (0);
}
#endif
