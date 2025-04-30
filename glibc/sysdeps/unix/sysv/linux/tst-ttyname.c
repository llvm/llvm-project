/* Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <unistd.h>

#include <support/check.h>
#include <support/namespace.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <support/test-driver.h>
#include <support/xunistd.h>

/* generic utilities */

#define VERIFY(expr)                                                    \
  do {                                                                  \
    if (!(expr))                                                        \
      {                                                                 \
        printf ("error: %s:%d: %s: %m\n",                               \
                __FILE__, __LINE__, #expr);                             \
        exit (1);                                                       \
      }                                                                 \
  } while (0)

static void
touch (const char *path, mode_t mode)
{
  xclose (xopen (path, O_WRONLY|O_CREAT|O_NOCTTY, mode));
}

static size_t
trim_prefix (char *str, size_t str_len, const char *prefix)
{
  size_t prefix_len = strlen (prefix);
  if (str_len > prefix_len && memcmp (str, prefix, prefix_len) == 0)
    {
      memmove (str, str + prefix_len, str_len - prefix_len);
      return str_len - prefix_len;
    }
  return str_len;
}

/* returns a pointer to static storage */
static char *
proc_fd_readlink (const char *linkname)
{
  static char target[PATH_MAX+1];
  ssize_t target_len = readlink (linkname, target, PATH_MAX);
  VERIFY (target_len > 0);
  target_len = trim_prefix (target, target_len, "(unreachable)");
  target[target_len] = '\0';
  return target;
}

/* plain ttyname runner */

struct result
{
  const char *name;
  int err;
};

/* strings in result structure are in static storage */
static struct result
run_ttyname (int fd)
{
  struct result ret;
  errno = 0;
  ret.name = ttyname (fd);
  ret.err = errno;
  return ret;
}

static bool
eq_ttyname (struct result actual, struct result expected)
{
  char *actual_name, *expected_name;

  if ((actual.err == expected.err)
      && (!actual.name == !expected.name)
      && (actual.name ? strcmp (actual.name, expected.name) == 0 : true))
    {
      if (expected.name)
        expected_name = xasprintf ("\"%s\"", expected.name);
      else
	expected_name = xstrdup ("NULL");

      printf ("info:      ttyname: PASS {name=%s, errno=%d}\n",
	      expected_name, expected.err);

      free (expected_name);
      return true;
    }

  if (actual.name)
    actual_name = xasprintf ("\"%s\"", actual.name);
  else
    actual_name = xstrdup ("NULL");

  if (expected.name)
    expected_name = xasprintf ("\"%s\"", expected.name);
  else
    expected_name = xstrdup ("NULL");

  printf ("error:     ttyname: actual {name=%s, errno=%d} != expected {name=%s, errno=%d}\n",
	  actual_name, actual.err,
	  expected_name, expected.err);

  free (actual_name);
  free (expected_name);
  return false;
}

/* ttyname_r runner */

struct result_r
{
  const char *name;
  int ret;
  int err;
};

/* strings in result structure are in static storage */
static struct result_r
run_ttyname_r (int fd)
{
  static char buf[TTY_NAME_MAX];

  struct result_r ret;
  errno = 0;
  ret.ret = ttyname_r (fd, buf, TTY_NAME_MAX);
  ret.err = errno;
  if (ret.ret == 0)
    ret.name = buf;
  else
    ret.name = NULL;
  return ret;
}

static bool
eq_ttyname_r (struct result_r actual, struct result_r expected)
{
  char *actual_name, *expected_name;

  if ((actual.err == expected.err)
      && (actual.ret == expected.ret)
      && (!actual.name == !expected.name)
      && (actual.name ? strcmp (actual.name, expected.name) == 0 : true))
    {
      if (expected.name)
        expected_name = xasprintf ("\"%s\"", expected.name);
      else
        expected_name = xstrdup ("NULL");

      printf ("info:      ttyname_r: PASS {name=%s, ret=%d, errno=%d}\n",
              expected_name, expected.ret, expected.err);

      free (expected_name);
      return true;
    }

  if (actual.name)
    actual_name = xasprintf ("\"%s\"", actual.name);
  else
    actual_name = xstrdup ("NULL");

  if (expected.name)
    expected_name = xasprintf ("\"%s\"", expected.name);
  else
    expected_name = xstrdup ("NULL");

  printf ("error:     ttyname_r: actual {name=%s, ret=%d, errno=%d} != expected {name=%s, ret=%d, errno=%d}\n",
	  actual_name, actual.ret, actual.err,
	  expected_name, expected.ret, expected.err);

  free (actual_name);
  free (expected_name);
  return false;
}

/* combined runner */

static bool
doit (int fd, const char *testname, struct result_r expected_r)
{
  struct result expected = {.name=expected_r.name, .err=expected_r.ret};
  bool ret = true;

  printf ("info:    testcase: %s\n", testname);

  if (!eq_ttyname (run_ttyname (fd), expected))
    ret = false;
  if (!eq_ttyname_r (run_ttyname_r (fd), expected_r))
    ret = false;

  if (!ret)
    support_record_failure ();

  return ret;
}

/* chroot setup */

static char *chrootdir;

static void
prepare (int argc, char **argv)
{
  chrootdir = xasprintf ("%s/tst-ttyname-XXXXXX", test_dir);
  if (mkdtemp (chrootdir) == NULL)
    FAIL_EXIT1 ("mkdtemp (\"%s\"): %m", chrootdir);
  add_temp_file (chrootdir);
}
#define PREPARE prepare

/* Adjust the file limit so that we have a chance to open PTY.  */
static void
adjust_file_limit (const char *pty)
{
  int number = -1;
  if (sscanf (pty, "/dev/pts/%d", &number) != 1 || number < 0)
    FAIL_EXIT1 ("invalid PTY name: \"%s\"", pty);

  /* Add a few additional descriptors to cover standard I/O streams
     etc.  */
  rlim_t desired_limit = number + 10;

  struct rlimit lim;
  if (getrlimit (RLIMIT_NOFILE, &lim) != 0)
    FAIL_EXIT1 ("getrlimit (RLIMIT_NOFILE): %m");
  if (lim.rlim_cur < desired_limit)
    {
      printf ("info: adjusting RLIMIT_NOFILE from %llu to %llu\n",
	      (unsigned long long int) lim.rlim_cur,
	      (unsigned long long int) desired_limit);
      lim.rlim_cur = desired_limit;
      if (setrlimit (RLIMIT_NOFILE, &lim) != 0)
	printf ("warning: setrlimit (RLIMIT_NOFILE) failed: %m\n");
    }
}

/* These chroot setup functions put the TTY at at "/console" (where it
   won't be found by ttyname), and create "/dev/console" as an
   ordinary file.  This way, it's easier to write test-cases that
   expect ttyname to fail; test-cases that expect it to succeed need
   to explicitly remount it at "/dev/console".  */

static int
do_in_chroot_1 (int (*cb)(const char *, int))
{
  printf ("info:  entering chroot 1\n");

  /* Open the PTS that we'll be testing on.  */
  int master;
  char *slavename;
  master = posix_openpt (O_RDWR|O_NOCTTY|O_NONBLOCK);
  if (master < 0)
    {
      if (errno == ENOENT)
	FAIL_UNSUPPORTED ("posix_openpt: %m");
      else
	FAIL_EXIT1 ("posix_openpt: %m");
    }
  VERIFY ((slavename = ptsname (master)));
  VERIFY (unlockpt (master) == 0);
  if (strncmp (slavename, "/dev/pts/", 9) != 0)
    FAIL_UNSUPPORTED ("slave pseudo-terminal is not under /dev/pts/: %s",
                      slavename);
  adjust_file_limit (slavename);
  int slave = xopen (slavename, O_RDWR, 0);
  if (!doit (slave, "basic smoketest",
             (struct result_r){.name=slavename, .ret=0, .err=0}))
    return 1;

  pid_t pid = xfork ();
  if (pid == 0)
    {
      xclose (master);

      if (!support_enter_mount_namespace ())
	FAIL_UNSUPPORTED ("could not enter new mount namespace");

      VERIFY (mount ("tmpfs", chrootdir, "tmpfs", 0, "mode=755") == 0);
      VERIFY (chdir (chrootdir) == 0);

      xmkdir ("proc", 0755);
      xmkdir ("dev", 0755);
      xmkdir ("dev/pts", 0755);

      VERIFY (mount ("/proc", "proc", NULL, MS_BIND|MS_REC, NULL) == 0);
      VERIFY (mount ("devpts", "dev/pts", "devpts",
                     MS_NOSUID|MS_NOEXEC,
                     "newinstance,ptmxmode=0666,mode=620") == 0);
      VERIFY (symlink ("pts/ptmx", "dev/ptmx") == 0);

      touch ("console", 0);
      touch ("dev/console", 0);
      VERIFY (mount (slavename, "console", NULL, MS_BIND, NULL) == 0);

      xchroot (".");

      char *linkname = xasprintf ("/proc/self/fd/%d", slave);
      char *target = proc_fd_readlink (linkname);
      VERIFY (strcmp (target, slavename) == 0);
      free (linkname);

      _exit (cb (slavename, slave));
    }
  int status;
  xwaitpid (pid, &status, 0);
  VERIFY (WIFEXITED (status));
  xclose (master);
  xclose (slave);
  return WEXITSTATUS (status);
}

static int
do_in_chroot_2 (int (*cb)(const char *, int))
{
  printf ("info:  entering chroot 2\n");

  int pid_pipe[2];
  xpipe (pid_pipe);
  int exit_pipe[2];
  xpipe (exit_pipe);

  /* Open the PTS that we'll be testing on.  */
  int master;
  char *slavename;
  VERIFY ((master = posix_openpt (O_RDWR|O_NOCTTY|O_NONBLOCK)) >= 0);
  VERIFY ((slavename = ptsname (master)));
  VERIFY (unlockpt (master) == 0);
  if (strncmp (slavename, "/dev/pts/", 9) != 0)
    FAIL_UNSUPPORTED ("slave pseudo-terminal is not under /dev/pts/: %s",
                      slavename);
  adjust_file_limit (slavename);
  /* wait until in a new mount ns to open the slave */

  /* enable `wait`ing on grandchildren */
  VERIFY (prctl (PR_SET_CHILD_SUBREAPER, 1) == 0);

  pid_t pid = xfork (); /* outer child */
  if (pid == 0)
    {
      xclose (master);
      xclose (pid_pipe[0]);
      xclose (exit_pipe[1]);

      if (!support_enter_mount_namespace ())
	FAIL_UNSUPPORTED ("could not enter new mount namespace");

      int slave = xopen (slavename, O_RDWR, 0);
      if (!doit (slave, "basic smoketest",
                 (struct result_r){.name=slavename, .ret=0, .err=0}))
        _exit (1);

      VERIFY (mount ("tmpfs", chrootdir, "tmpfs", 0, "mode=755") == 0);
      VERIFY (chdir (chrootdir) == 0);

      xmkdir ("proc", 0755);
      xmkdir ("dev", 0755);
      xmkdir ("dev/pts", 0755);

      VERIFY (mount ("devpts", "dev/pts", "devpts",
                     MS_NOSUID|MS_NOEXEC,
                     "newinstance,ptmxmode=0666,mode=620") == 0);
      VERIFY (symlink ("pts/ptmx", "dev/ptmx") == 0);

      touch ("console", 0);
      touch ("dev/console", 0);
      VERIFY (mount (slavename, "console", NULL, MS_BIND, NULL) == 0);

      xchroot (".");

      if (unshare (CLONE_NEWNS | CLONE_NEWPID) < 0)
        FAIL_UNSUPPORTED ("could not enter new PID namespace");
      pid = xfork (); /* inner child */
      if (pid == 0)
        {
          xclose (pid_pipe[1]);

          /* wait until the outer child has exited */
          char c;
          VERIFY (read (exit_pipe[0], &c, 1) == 0);
          xclose (exit_pipe[0]);

          VERIFY (mount ("proc", "/proc", "proc",
                         MS_NOSUID|MS_NOEXEC|MS_NODEV, NULL) == 0);

          char *linkname = xasprintf ("/proc/self/fd/%d", slave);
          char *target = proc_fd_readlink (linkname);
          VERIFY (strcmp (target, strrchr (slavename, '/')) == 0);
          free (linkname);

          _exit (cb (slavename, slave));
        }
      xwrite (pid_pipe[1], &pid, sizeof pid);
      _exit (0);
    }
  xclose (pid_pipe[1]);
  xclose (exit_pipe[0]);
  xclose (exit_pipe[1]);

  /* wait for the outer child */
  int status;
  xwaitpid (pid, &status, 0);
  VERIFY (WIFEXITED (status));
  int ret = WEXITSTATUS (status);
  if (ret != 0)
    return ret;

  /* set 'pid' to the inner child */
  VERIFY (read (pid_pipe[0], &pid, sizeof pid) == sizeof pid);
  xclose (pid_pipe[0]);

  /* wait for the inner child */
  xwaitpid (pid, &status, 0);
  VERIFY (WIFEXITED (status));
  xclose (master);
  return WEXITSTATUS (status);
}

/* main test */

static int
run_chroot_tests (const char *slavename, int slave)
{
  struct stat st;
  bool ok = true;

  /* There are 3 groups of tests here.  The first group fairly
     generically does things known to mess up ttyname, and verifies
     that ttyname copes correctly.  The remaining groups are
     increasingly convoluted, as we target specific parts of ttyname
     to try to confuse.  */

  /* Basic tests that it doesn't get confused by multiple devpts
     instances.  */
  {
    VERIFY (stat (slavename, &st) < 0); /* sanity check */
    if (!doit (slave, "no conflict, no match",
               (struct result_r){.name=NULL, .ret=ENODEV, .err=ENODEV}))
      ok = false;
    VERIFY (mount ("/console", "/dev/console", NULL, MS_BIND, NULL) == 0);
    if (!doit (slave, "no conflict, console",
               (struct result_r){.name="/dev/console", .ret=0, .err=0}))
      ok = false;
    VERIFY (umount ("/dev/console") == 0);

    /* Keep creating PTYs until we we get a name collision.  */
    while (true)
      {
	if (stat (slavename, &st) == 0)
	  break;
	if (posix_openpt (O_RDWR|O_NOCTTY|O_NONBLOCK) < 0)
	  {
	    if (errno == ENOSPC || errno == EMFILE || errno == ENFILE)
	      FAIL_UNSUPPORTED ("cannot re-create PTY \"%s\" in chroot: %m"
				" (consider increasing limits)", slavename);
	    else
	      FAIL_EXIT1 ("cannot re-create PTY \"%s\" chroot: %m", slavename);
	  }
      }

    if (!doit (slave, "conflict, no match",
               (struct result_r){.name=NULL, .ret=ENODEV, .err=ENODEV}))
      ok = false;
    VERIFY (mount ("/console", "/dev/console", NULL, MS_BIND, NULL) == 0);
    if (!doit (slave, "conflict, console",
               (struct result_r){.name="/dev/console", .ret=0, .err=0}))
      ok = false;
    VERIFY (umount ("/dev/console") == 0);
  }

  /* The first tests kinda assumed that they hit certain code-paths
     based on assuming that the readlink target is 'slavename', but
     that's not quite always true.  They're still a good preliminary
     sanity check, so keep them, but let's add tests that make sure
     that those code-paths are hit by doing a readlink ourself.  */
  {
    char *linkname = xasprintf ("/proc/self/fd/%d", slave);
    char *target = proc_fd_readlink (linkname);
    free (linkname);
    /* Depeding on how we set up the chroot, the kernel may or may not
       trim the leading path to the target (it may give us "/6",
       instead of "/dev/pts/6").  We test it both ways (do_in_chroot_1
       and do_in_chroot_2).  This test group relies on the target
       existing, so guarantee that it does exist by creating it if
       necessary.  */
    if (stat (target, &st) < 0)
      {
        VERIFY (errno == ENOENT);
        touch (target, 0);
      }

    VERIFY (mount ("/console", "/dev/console", NULL, MS_BIND, NULL) == 0);
    VERIFY (mount ("/console", target, NULL, MS_BIND, NULL) == 0);
    if (!doit (slave, "with readlink target",
               (struct result_r){.name=target, .ret=0, .err=0}))
      ok = false;
    VERIFY (umount (target) == 0);
    VERIFY (umount ("/dev/console") == 0);

    VERIFY (mount ("/console", "/dev/console", NULL, MS_BIND, NULL) == 0);
    VERIFY (mount (slavename, target, NULL, MS_BIND, NULL) == 0);
    if (!doit (slave, "with readlink trap; fallback",
               (struct result_r){.name="/dev/console", .ret=0, .err=0}))
      ok = false;
    VERIFY (umount (target) == 0);
    VERIFY (umount ("/dev/console") == 0);

    VERIFY (mount (slavename, target, NULL, MS_BIND, NULL) == 0);
    if (!doit (slave, "with readlink trap; no fallback",
               (struct result_r){.name=NULL, .ret=ENODEV, .err=ENODEV}))
      ok = false;
    VERIFY (umount (target) == 0);
  }

  /* This test makes sure that everything still works OK if readdir
     finds a pseudo-match before and/or after the actual match.  Now,
     to do that, we need to control that readdir finds the
     pseudo-matches before and after the actual match; and there's no
     good way to control that order in absence of whitebox testing.
     So, just create 3 files, then use opendir/readdir to see what
     order they are in, and assign meaning based on that order, not by
     name; assigning the first to be a pseudo-match, the second to be
     the actual match, and the third to be a pseudo-match.  This
     assumes that (on tmpfs) ordering within the directory is stable
     in the absence of modification, which seems reasonably safe.  */
  {
    /* since we're testing the fallback search, disable the readlink
       happy-path */
    VERIFY (umount2 ("/proc", MNT_DETACH) == 0);

    touch ("/dev/console1", 0);
    touch ("/dev/console2", 0);
    touch ("/dev/console3", 0);

    char *c[3];
    int ci = 0;
    DIR *dirstream = opendir ("/dev");
    VERIFY (dirstream != NULL);
    struct dirent *d;
    while ((d = readdir (dirstream)) != NULL && ci < 3)
      {
        if (strcmp (d->d_name, "console1")
            && strcmp (d->d_name, "console2")
            && strcmp (d->d_name, "console3") )
          continue;
        c[ci++] = xasprintf ("/dev/%s", d->d_name);
      }
    VERIFY (ci == 3);
    VERIFY (closedir (dirstream) == 0);

    VERIFY (mount (slavename, c[0], NULL, MS_BIND, NULL) == 0);
    VERIFY (mount ("/console", c[1], NULL, MS_BIND, NULL) == 0);
    VERIFY (mount (slavename, c[2], NULL, MS_BIND, NULL) == 0);
    VERIFY (umount2 ("/dev/pts", MNT_DETACH) == 0);
    if (!doit (slave, "with search-path trap",
               (struct result_r){.name=c[1], .ret=0, .err=0}))
      ok = false;
    for (int i = 0; i < 3; i++)
      {
        VERIFY (umount (c[i]) == 0);
        VERIFY (unlink (c[i]) == 0);
        free (c[i]);
      }
  }

  return ok ? 0 : 1;
}

static int
do_test (void)
{
  support_become_root ();

  int ret1 = do_in_chroot_1 (run_chroot_tests);
  if (ret1 == EXIT_UNSUPPORTED)
    return ret1;

  int ret2 = do_in_chroot_2 (run_chroot_tests);
  if (ret2 == EXIT_UNSUPPORTED)
    return ret2;

  return  ret1 | ret2;
}

#include <support/test-driver.c>
