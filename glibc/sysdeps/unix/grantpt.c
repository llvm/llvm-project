/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Zack Weinberg <zack@rabi.phys.columbia.edu>, 1998.

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

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <grp.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "pty-private.h"


/* Return the result of ptsname_r in the buffer pointed to by PTS,
   which should be of length BUF_LEN.  If it is too long to fit in
   this buffer, a sufficiently long buffer is allocated using malloc,
   and returned in PTS.  0 is returned upon success, -1 otherwise.  */
static int
pts_name (int fd, char **pts, size_t buf_len, struct stat64 *stp)
{
  int rv;
  char *buf = *pts;

  for (;;)
    {
      char *new_buf;

      if (buf_len)
	{
	  rv = __ptsname_internal (fd, buf, buf_len, stp);
	  if (rv != 0)
	    {
	      if (rv == ENOTTY)
		/* ptsname_r returns with ENOTTY to indicate
		   a descriptor not referring to a pty master.
		   For this condition, grantpt must return EINVAL.  */
		rv = EINVAL;
	      errno = rv;	/* Not necessarily set by __ptsname_r.  */
	      break;
	    }

	  if (memchr (buf, '\0', buf_len))
	    /* We succeeded and the returned name fit in the buffer.  */
	    break;

	  /* Try again with a longer buffer.  */
	  buf_len += buf_len;	/* Double it */
	}
      else
	/* No initial buffer; start out by mallocing one.  */
	buf_len = 128;		/* First time guess.  */

      if (buf != *pts)
	/* We've already malloced another buffer at least once.  */
	new_buf = (char *) realloc (buf, buf_len);
      else
	new_buf = (char *) malloc (buf_len);
      if (! new_buf)
	{
	  rv = -1;
	  __set_errno (ENOMEM);
	  break;
	}
      buf = new_buf;
    }

  if (rv == 0)
    *pts = buf;		/* Return buffer to the user.  */
  else if (buf != *pts)
    free (buf);		/* Free what we malloced when returning an error.  */

  return rv;
}

/* Change the ownership and access permission of the slave pseudo
   terminal associated with the master pseudo terminal specified
   by FD.  */
int
grantpt (int fd)
{
  int retval = -1;
#ifdef PATH_MAX
  char _buf[PATH_MAX];
#else
  char _buf[512];
#endif
  char *buf = _buf;
  struct stat64 st;

  if (__glibc_unlikely (pts_name (fd, &buf, sizeof (_buf), &st)))
    {
      int save_errno = errno;

      /* Check, if the file descriptor is valid.  pts_name returns the
	 wrong errno number, so we cannot use that.  */
      if (__libc_fcntl (fd, F_GETFD) == -1 && errno == EBADF)
	return -1;

       /* If the filedescriptor is no TTY, grantpt has to set errno
	  to EINVAL.  */
       if (save_errno == ENOTTY)
	 __set_errno (EINVAL);
       else
	 __set_errno (save_errno);

       return -1;
    }

  /* Make sure that we own the device.  */
  uid_t uid = __getuid ();
  if (st.st_uid != uid)
    {
      if (__chown (buf, uid, st.st_gid) < 0)
	goto helper;
    }

  static int tty_gid = -1;
  if (__glibc_unlikely (tty_gid == -1))
    {
      char *grtmpbuf;
      struct group grbuf;
      size_t grbuflen = __sysconf (_SC_GETGR_R_SIZE_MAX);
      struct group *p;

      /* Get the group ID of the special `tty' group.  */
      if (grbuflen == (size_t) -1L)
	/* `sysconf' does not support _SC_GETGR_R_SIZE_MAX.
	   Try a moderate value.  */
	grbuflen = 1024;
      grtmpbuf = (char *) __alloca (grbuflen);
      __getgrnam_r (TTY_GROUP, &grbuf, grtmpbuf, grbuflen, &p);
      if (p != NULL)
	tty_gid = p->gr_gid;
    }
  gid_t gid = tty_gid == -1 ? __getgid () : tty_gid;

#if HAVE_PT_CHOWN
  /* Make sure the group of the device is that special group.  */
  if (st.st_gid != gid)
    {
      if (__chown (buf, uid, gid) < 0)
	goto helper;
    }

  /* Make sure the permission mode is set to readable and writable by
     the owner, and writable by the group.  */
  mode_t mode = S_IRUSR|S_IWUSR|S_IWGRP;
#else
  /* When built without pt_chown, we have delegated the creation of the
     pty node with the right group and permission mode to the kernel, and
     non-root users are unlikely to be able to change it. Therefore let's
     consider that POSIX enforcement is the responsibility of the whole
     system and not only the GNU libc. Thus accept different group or
     permission mode.  */

  /* Make sure the permission is set to readable and writable by the
     owner.  For security reasons, make it writable by the group only
     when originally writable and when the group of the device is that
     special group.  */
  mode_t mode = S_IRUSR|S_IWUSR
	        |((st.st_gid == gid) ? (st.st_mode & S_IWGRP) : 0);
#endif

  if ((st.st_mode & ACCESSPERMS) != mode)
    {
      if (__chmod (buf, mode) < 0)
	goto helper;
    }

  retval = 0;
  goto cleanup;

  /* We have to use the helper program if it is available.  */
 helper:;

#if HAVE_PT_CHOWN
  pid_t pid = __fork ();
  if (pid == -1)
    goto cleanup;
  else if (pid == 0)
    {
      /* Disable core dumps.  */
      struct rlimit rl = { 0, 0 };
      __setrlimit (RLIMIT_CORE, &rl);

      /* We pass the master pseudo terminal as file descriptor PTY_FILENO.  */
      if (fd != PTY_FILENO)
	if (__dup2 (fd, PTY_FILENO) < 0)
	  _exit (FAIL_EBADF);

# ifdef CLOSE_ALL_FDS
      CLOSE_ALL_FDS ();
# endif

      execle (_PATH_PT_CHOWN, __basename (_PATH_PT_CHOWN), NULL, NULL);
      _exit (FAIL_EXEC);
    }
  else
    {
      int w;

      if (__waitpid (pid, &w, 0) == -1)
	goto cleanup;
      if (!WIFEXITED (w))
	__set_errno (ENOEXEC);
      else
	switch (WEXITSTATUS (w))
	  {
	  case 0:
	    retval = 0;
	    break;
	  case FAIL_EBADF:
	    __set_errno (EBADF);
	    break;
	  case FAIL_EINVAL:
	    __set_errno (EINVAL);
	    break;
	  case FAIL_EACCES:
	    __set_errno (EACCES);
	    break;
	  case FAIL_EXEC:
	    __set_errno (ENOEXEC);
	    break;
	  case FAIL_ENOMEM:
	    __set_errno (ENOMEM);
	    break;

	  default:
	    assert(! "grantpt: internal error: invalid exit code from pt_chown");
	  }
    }
#endif

 cleanup:
  if (buf != _buf)
    free (buf);

  return retval;
}
libc_hidden_def (grantpt)
