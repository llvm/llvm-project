/* Capture output from a subprocess.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <support/subprocess.h>
#include <support/capture_subprocess.h>

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <support/check.h>
#include <support/xunistd.h>
#include <support/xsocket.h>
#include <support/xspawn.h>
#include <support/support.h>
#include <support/test-driver.h>

static void
transfer (const char *what, struct pollfd *pfd, struct xmemstream *stream)
{
  if (pfd->revents != 0)
    {
      char buf[1024];
      ssize_t ret = TEMP_FAILURE_RETRY (read (pfd->fd, buf, sizeof (buf)));
      if (ret < 0)
        {
          support_record_failure ();
          printf ("error: reading from subprocess %s: %m\n", what);
          pfd->events = 0;
          pfd->revents = 0;
        }
      else if (ret == 0)
        {
          /* EOF reached.  Stop listening.  */
          pfd->events = 0;
          pfd->revents = 0;
        }
      else
        /* Store the data just read.   */
        TEST_VERIFY (fwrite (buf, ret, 1, stream->out) == 1);
    }
}

static void
support_capture_poll (struct support_capture_subprocess *result,
		      struct support_subprocess *proc)
{
  struct pollfd fds[2] =
    {
      { .fd = proc->stdout_pipe[0], .events = POLLIN },
      { .fd = proc->stderr_pipe[0], .events = POLLIN },
    };

  do
    {
      xpoll (fds, 2, -1);
      transfer ("stdout", &fds[0], &result->out);
      transfer ("stderr", &fds[1], &result->err);
    }
  while (fds[0].events != 0 || fds[1].events != 0);

  xfclose_memstream (&result->out);
  xfclose_memstream (&result->err);

  result->status = support_process_wait (proc);
}

struct support_capture_subprocess
support_capture_subprocess (void (*callback) (void *), void *closure)
{
  struct support_capture_subprocess result;
  xopen_memstream (&result.out);
  xopen_memstream (&result.err);

  struct support_subprocess proc = support_subprocess (callback, closure);

  support_capture_poll (&result, &proc);
  return result;
}

struct support_capture_subprocess
support_capture_subprogram (const char *file, char *const argv[])
{
  struct support_capture_subprocess result;
  xopen_memstream (&result.out);
  xopen_memstream (&result.err);

  struct support_subprocess proc = support_subprogram (file, argv);

  support_capture_poll (&result, &proc);
  return result;
}

/* Copies the executable into a restricted directory, so that we can
   safely make it SGID with the TARGET group ID.  Then runs the
   executable.  */
static int
copy_and_spawn_sgid (char *child_id, gid_t gid)
{
  char *dirname = xasprintf ("%s/tst-tunables-setuid.%jd",
			     test_dir, (intmax_t) getpid ());
  char *execname = xasprintf ("%s/bin", dirname);
  int infd = -1;
  int outfd = -1;
  int ret = 1, status = 1;

  TEST_VERIFY (mkdir (dirname, 0700) == 0);
  if (support_record_failure_is_failed ())
    goto err;

  infd = open ("/proc/self/exe", O_RDONLY);
  if (infd < 0)
    FAIL_UNSUPPORTED ("unsupported: Cannot read binary from procfs\n");

  outfd = open (execname, O_WRONLY | O_CREAT | O_EXCL, 0700);
  TEST_VERIFY (outfd >= 0);
  if (support_record_failure_is_failed ())
    goto err;

  char buf[4096];
  for (;;)
    {
      ssize_t rdcount = read (infd, buf, sizeof (buf));
      TEST_VERIFY (rdcount >= 0);
      if (support_record_failure_is_failed ())
	goto err;
      if (rdcount == 0)
	break;
      char *p = buf;
      char *end = buf + rdcount;
      while (p != end)
	{
	  ssize_t wrcount = write (outfd, buf, end - p);
	  if (wrcount == 0)
	    errno = ENOSPC;
	  TEST_VERIFY (wrcount > 0);
	  if (support_record_failure_is_failed ())
	    goto err;
	  p += wrcount;
	}
    }
  TEST_VERIFY (fchown (outfd, getuid (), gid) == 0);
  if (support_record_failure_is_failed ())
    goto err;
  TEST_VERIFY (fchmod (outfd, 02750) == 0);
  if (support_record_failure_is_failed ())
    goto err;
  TEST_VERIFY (close (outfd) == 0);
  if (support_record_failure_is_failed ())
    goto err;
  TEST_VERIFY (close (infd) == 0);
  if (support_record_failure_is_failed ())
    goto err;

  /* We have the binary, now spawn the subprocess.  Avoid using
     support_subprogram because we only want the program exit status, not the
     contents.  */
  ret = 0;

  char * const args[] = {execname, child_id, NULL};

  status = support_subprogram_wait (args[0], args);

err:
  if (outfd >= 0)
    close (outfd);
  if (infd >= 0)
    close (infd);
  if (execname != NULL)
    {
      unlink (execname);
      free (execname);
    }
  if (dirname != NULL)
    {
      rmdir (dirname);
      free (dirname);
    }

  if (ret != 0)
    FAIL_EXIT1("Failed to make sgid executable for test\n");

  return status;
}

int
support_capture_subprogram_self_sgid (char *child_id)
{
  gid_t target = 0;
  const int count = 64;
  gid_t groups[count];

  /* Get a GID which is not our current GID, but is present in the
     supplementary group list.  */
  int ret = getgroups (count, groups);
  if (ret < 0)
    FAIL_UNSUPPORTED("Could not get group list for user %jd\n",
		     (intmax_t) getuid ());

  gid_t current = getgid ();
  for (int i = 0; i < ret; ++i)
    {
      if (groups[i] != current)
	{
	  target = groups[i];
	  break;
	}
    }

  if (target == 0)
    FAIL_UNSUPPORTED("Could not find a suitable GID for user %jd\n",
		     (intmax_t) getuid ());

  return copy_and_spawn_sgid (child_id, target);
}

void
support_capture_subprocess_free (struct support_capture_subprocess *p)
{
  free (p->out.buffer);
  free (p->err.buffer);
}
