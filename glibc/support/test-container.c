/* Run a test case in an isolated namespace.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#define _FILE_OFFSET_BITS 64

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/fcntl.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <stdarg.h>
#include <sys/sysmacros.h>
#include <ctype.h>
#include <utime.h>
#include <errno.h>
#include <error.h>
#include <libc-pointer-arith.h>

#ifdef __linux__
#include <sys/mount.h>
#endif

#include <support/support.h>
#include <support/xunistd.h>
#include <support/capture_subprocess.h>
#include "check.h"
#include "test-driver.h"

#ifndef __linux__
#define mount(s,t,fs,f,d) no_mount()
int no_mount (void)
{
  FAIL_UNSUPPORTED("mount not supported; port needed");
}
#endif

int verbose = 0;

/* Running a test in a container is tricky.  There are two main
   categories of things to do:

   1. "Once" actions, like setting up the container and doing an
      install into it.

   2. "Per-test" actions, like copying in support files and
      configuring the container.


   "Once" actions:

   * mkdir $buildroot/testroot.pristine/
   * install into it
     * default glibc install
     * create /bin for /bin/sh
     * create $(complocaledir) so localedef tests work with default paths.
     * install /bin/sh, /bin/echo, and /bin/true.
   * rsync to $buildroot/testroot.root/

   "Per-test" actions:
   * maybe rsync to $buildroot/testroot.root/
   * copy support files and test binary
   * chroot/unshare
   * set up any mounts (like /proc)
   * run ldconfig

   Magic files:

   For test $srcdir/foo/mytest.c we look for $srcdir/foo/mytest.root
   and, if found...

   * mytest.root/ is rsync'd into container
   * mytest.root/preclean.req causes fresh rsync (with delete) before
     test if present
   * mytest.root/mytest.script has a list of "commands" to run:
       syntax:
         # comment
         su
         mv FILE FILE
	 cp FILE FILE
	 rm FILE
	 cwd PATH
	 exec FILE
	 mkdirp MODE DIR

       variables:
	 $B/ build dir, equivalent to $(common-objpfx)
	 $S/ source dir, equivalent to $(srcdir)
	 $I/ install dir, equivalent to $(prefix)
	 $L/ library dir (in container), equivalent to $(libdir)
	 $complocaledir/ compiled locale dir, equivalent to $(complocaledir)
	 / container's root

	 If FILE begins with any of these variables then they will be
	 substituted for the described value.

	 The goal is to expose as many of the runtime's configured paths
	 via variables so they can be used to setup the container environment
	 before execution reaches the test.

       details:
         - '#': A comment.
         - 'su': Enables running test as root in the container.
         - 'mv': A minimal move files command.
         - 'cp': A minimal copy files command.
         - 'rm': A minimal remove files command.
	 - 'cwd': set test working directory
	 - 'exec': change test binary location (may end in /)
	 - 'mkdirp': A minimal "mkdir -p FILE" command.

   * mytest.root/postclean.req causes fresh rsync (with delete) after
     test if present

   * mytest.root/ldconfig.run causes ldconfig to be issued prior
     test execution (to setup the initial ld.so.cache).

   Note that $srcdir/foo/mytest.script may be used instead of a
   $srcdir/foo/mytest.root/mytest.script in the sysroot template, if
   there is no other reason for a sysroot.

   Design goals:

   * independent of other packages which may not be installed (like
     rsync or Docker, or even "cp")

   * Simple, easy to review code (i.e. prefer simple naive code over
     complex efficient code)

   * The current implementation ist parallel-make-safe, but only in
     that it uses a lock to prevent parallel access to the testroot.  */


/* Utility Functions */

/* Like xunlink, but it's OK if the file already doesn't exist.  */
void
maybe_xunlink (const char *path)
{
  int rv = unlink (path);
  if (rv < 0 && errno != ENOENT)
    FAIL_EXIT1 ("unlink (\"%s\"): %m", path);
}

/* Like xmkdir, but it's OK if the directory already exists.  */
void
maybe_xmkdir (const char *path, mode_t mode)
{
  struct stat st;

  if (stat (path, &st) == 0
      && S_ISDIR (st.st_mode))
    return;
  xmkdir (path, mode);
}

/* Temporarily concatenate multiple strings into one.  Allows up to 10
   temporary results; use xstrdup () if you need them to be
   permanent.  */
static char *
concat (const char *str, ...)
{
  /* Assume initialized to NULL/zero.  */
  static char *bufs[10];
  static size_t buflens[10];
  static int bufn = 0;
  int n;
  size_t len;
  va_list ap, ap2;
  char *cp;
  char *next;

  va_start (ap, str);
  va_copy (ap2, ap);

  n = bufn;
  bufn = (bufn + 1) % 10;
  len = strlen (str);

  while ((next = va_arg (ap, char *)) != NULL)
    len = len + strlen (next);

  va_end (ap);

  if (bufs[n] == NULL)
    {
      bufs[n] = xmalloc (len + 1); /* NUL */
      buflens[n] = len + 1;
    }
  else if (buflens[n] < len + 1)
    {
      bufs[n] = xrealloc (bufs[n], len + 1); /* NUL */
      buflens[n] = len + 1;
    }

  strcpy (bufs[n], str);
  cp = strchr (bufs[n], '\0');
  while ((next = va_arg (ap2, char *)) != NULL)
    {
      strcpy (cp, next);
      cp = strchr (cp, '\0');
    }
  *cp = 0;
  va_end (ap2);

  return bufs[n];
}

/* Try to mount SRC onto DEST.  */
static void
trymount (const char *src, const char *dest)
{
  if (mount (src, dest, "", MS_BIND, NULL) < 0)
    FAIL_EXIT1 ("can't mount %s onto %s\n", src, dest);
}

/* Special case of above for devices like /dev/zero where we have to
   mount a device over a device, not a directory over a directory.  */
static void
devmount (const char *new_root_path, const char *which)
{
  int fd;
  fd = open (concat (new_root_path, "/dev/", which, NULL),
	     O_CREAT | O_TRUNC | O_RDWR, 0777);
  xclose (fd);

  trymount (concat ("/dev/", which, NULL),
	    concat (new_root_path, "/dev/", which, NULL));
}

/* Returns true if the string "looks like" an environement variable
   being set.  */
static int
is_env_setting (const char *a)
{
  int count_name = 0;

  while (*a)
    {
      if (isalnum (*a) || *a == '_')
	++count_name;
      else if (*a == '=' && count_name > 0)
	return 1;
      else
	return 0;
      ++a;
    }
  return 0;
}

/* Break the_line into words and store in the_words.  Max nwords,
   returns actual count.  */
static int
tokenize (char *the_line, char **the_words, int nwords)
{
  int rv = 0;

  while (nwords > 0)
    {
      /* Skip leading whitespace, if any.  */
      while (*the_line && isspace (*the_line))
	++the_line;

      /* End of line?  */
      if (*the_line == 0)
	return rv;

      /* THE_LINE points to a non-whitespace character, so we have a
	 word.  */
      *the_words = the_line;
      ++the_words;
      nwords--;
      ++rv;

      /* Skip leading whitespace, if any.  */
      while (*the_line && ! isspace (*the_line))
	++the_line;

      /* We now point at the trailing NUL *or* some whitespace.  */
      if (*the_line == 0)
	return rv;

      /* It was whitespace, skip and keep tokenizing.  */
      *the_line++ = 0;
    }

  /* We get here if we filled the words buffer.  */
  return rv;
}


/* Mini-RSYNC implementation.  Optimize later.      */

/* A few routines for an "rsync buffer" which stores the paths we're
   working on.  We continuously grow and shrink the paths in each
   buffer so there's lot of re-use.  */

/* We rely on "initialized to zero" to set these up.  */
typedef struct
{
  char *buf;
  size_t len;
  size_t size;
} path_buf;

static path_buf spath, dpath;

static void
r_setup (char *path, path_buf * pb)
{
  size_t len = strlen (path);
  if (pb->buf == NULL || pb->size < len + 1)
    {
      /* Round up.  This is an arbitrary number, just to keep from
	 reallocing too often.  */
      size_t sz = ALIGN_UP (len + 1, 512);
      if (pb->buf == NULL)
	pb->buf = (char *) xmalloc (sz);
      else
	pb->buf = (char *) xrealloc (pb->buf, sz);
      if (pb->buf == NULL)
	FAIL_EXIT1 ("Out of memory while rsyncing\n");

      pb->size = sz;
    }
  strcpy (pb->buf, path);
  pb->len = len;
}

static void
r_append (const char *path, path_buf * pb)
{
  size_t len = strlen (path) + pb->len;
  if (pb->size < len + 1)
    {
      /* Round up */
      size_t sz = ALIGN_UP (len + 1, 512);
      pb->buf = (char *) xrealloc (pb->buf, sz);
      if (pb->buf == NULL)
	FAIL_EXIT1 ("Out of memory while rsyncing\n");

      pb->size = sz;
    }
  strcpy (pb->buf + pb->len, path);
  pb->len = len;
}

static int
file_exists (char *path)
{
  struct stat st;
  if (lstat (path, &st) == 0)
    return 1;
  return 0;
}

static void
recursive_remove (char *path)
{
  pid_t child;
  int status;

  child = fork ();

  switch (child) {
  case -1:
    perror("fork");
    FAIL_EXIT1 ("Unable to fork");
  case 0:
    /* Child.  */
    execlp ("rm", "rm", "-rf", path, NULL);
    FAIL_EXIT1 ("exec rm: %m");
  default:
    /* Parent.  */
    waitpid (child, &status, 0);
    /* "rm" would have already printed a suitable error message.  */
    if (! WIFEXITED (status)
	|| WEXITSTATUS (status) != 0)
      FAIL_EXIT1 ("exec child returned status: %d", status);

    break;
  }
}

/* Used for both rsync and the mytest.script "cp" command.  */
static void
copy_one_file (const char *sname, const char *dname)
{
  int sfd, dfd;
  struct stat st;
  struct utimbuf times;

  sfd = open (sname, O_RDONLY);
  if (sfd < 0)
    FAIL_EXIT1 ("unable to open %s for reading\n", sname);

  if (fstat (sfd, &st) < 0)
    FAIL_EXIT1 ("unable to fstat %s\n", sname);

  dfd = open (dname, O_WRONLY | O_TRUNC | O_CREAT, 0600);
  if (dfd < 0)
    FAIL_EXIT1 ("unable to open %s for writing\n", dname);

  xcopy_file_range (sfd, 0, dfd, 0, st.st_size, 0);

  xclose (sfd);
  xclose (dfd);

  if (chmod (dname, st.st_mode & 0777) < 0)
    FAIL_EXIT1 ("chmod %s: %s\n", dname, strerror (errno));

  times.actime = st.st_atime;
  times.modtime = st.st_mtime;
  if (utime (dname, &times) < 0)
    FAIL_EXIT1 ("utime %s: %s\n", dname, strerror (errno));
}

/* We don't check *everything* about the two files to see if a copy is
   needed, just the minimum to make sure we get the latest copy.  */
static int
need_sync (char *ap, char *bp, struct stat *a, struct stat *b)
{
  if ((a->st_mode & S_IFMT) != (b->st_mode & S_IFMT))
    return 1;

  if (S_ISLNK (a->st_mode))
    {
      int rv;
      char *al, *bl;

      if (a->st_size != b->st_size)
	return 1;

      al = xreadlink (ap);
      bl = xreadlink (bp);
      rv = strcmp (al, bl);
      free (al);
      free (bl);
      if (rv == 0)
	return 0; /* links are same */
      return 1; /* links differ */
    }

  if (verbose)
    {
      if (a->st_size != b->st_size)
	printf ("SIZE\n");
      if ((a->st_mode & 0777) != (b->st_mode & 0777))
	printf ("MODE\n");
      if (a->st_mtime != b->st_mtime)
	printf ("TIME\n");
    }

  if (a->st_size == b->st_size
      && ((a->st_mode & 0777) == (b->st_mode & 0777))
      && a->st_mtime == b->st_mtime)
    return 0;

  return 1;
}

static void
rsync_1 (path_buf * src, path_buf * dest, int and_delete, int force_copies)
{
  DIR *dir;
  struct dirent *de;
  struct stat s, d;

  r_append ("/", src);
  r_append ("/", dest);

  if (verbose)
    printf ("sync %s to %s%s%s\n", src->buf, dest->buf,
	    and_delete ? " and delete" : "",
	    force_copies ? " (forced)" : "");

  size_t staillen = src->len;

  size_t dtaillen = dest->len;

  dir = opendir (src->buf);

  while ((de = readdir (dir)) != NULL)
    {
      if (strcmp (de->d_name, ".") == 0
	  || strcmp (de->d_name, "..") == 0)
	continue;

      src->len = staillen;
      r_append (de->d_name, src);
      dest->len = dtaillen;
      r_append (de->d_name, dest);

      s.st_mode = ~0;
      d.st_mode = ~0;

      if (lstat (src->buf, &s) != 0)
	FAIL_EXIT1 ("%s obtained by readdir, but stat failed.\n", src->buf);

      /* It's OK if this one fails, since we know the file might be
	 missing.  */
      lstat (dest->buf, &d);

      if (! force_copies && ! need_sync (src->buf, dest->buf, &s, &d))
	{
	  if (S_ISDIR (s.st_mode))
	    rsync_1 (src, dest, and_delete, force_copies);
	  continue;
	}

      if (d.st_mode != ~0)
	switch (d.st_mode & S_IFMT)
	  {
	  case S_IFDIR:
	    if (!S_ISDIR (s.st_mode))
	      {
		if (verbose)
		  printf ("-D %s\n", dest->buf);
		recursive_remove (dest->buf);
	      }
	    break;

	  default:
	    if (verbose)
	      printf ("-F %s\n", dest->buf);
	    maybe_xunlink (dest->buf);
	    break;
	  }

      switch (s.st_mode & S_IFMT)
	{
	case S_IFREG:
	  if (verbose)
	    printf ("+F %s\n", dest->buf);
	  copy_one_file (src->buf, dest->buf);
	  break;

	case S_IFDIR:
	  if (verbose)
	    printf ("+D %s\n", dest->buf);
	  maybe_xmkdir (dest->buf, (s.st_mode & 0777) | 0700);
	  rsync_1 (src, dest, and_delete, force_copies);
	  break;

	case S_IFLNK:
	  {
	    char *lp;
	    if (verbose)
	      printf ("+L %s\n", dest->buf);
	    lp = xreadlink (src->buf);
	    xsymlink (lp, dest->buf);
	    free (lp);
	    break;
	  }

	default:
	  break;
	}
    }

  closedir (dir);
  src->len = staillen;
  src->buf[staillen] = 0;
  dest->len = dtaillen;
  dest->buf[dtaillen] = 0;

  if (!and_delete)
    return;

  /* The rest of this function removes any files/directories in DEST
     that do not exist in SRC.  This is triggered as part of a
     preclean or postsclean step.  */

  dir = opendir (dest->buf);

  while ((de = readdir (dir)) != NULL)
    {
      if (strcmp (de->d_name, ".") == 0
	  || strcmp (de->d_name, "..") == 0)
	continue;

      src->len = staillen;
      r_append (de->d_name, src);
      dest->len = dtaillen;
      r_append (de->d_name, dest);

      s.st_mode = ~0;
      d.st_mode = ~0;

      lstat (src->buf, &s);

      if (lstat (dest->buf, &d) != 0)
	FAIL_EXIT1 ("%s obtained by readdir, but stat failed.\n", dest->buf);

      if (s.st_mode == ~0)
	{
	  /* dest exists and src doesn't, clean it.  */
	  switch (d.st_mode & S_IFMT)
	    {
	    case S_IFDIR:
	      if (!S_ISDIR (s.st_mode))
		{
		  if (verbose)
		    printf ("-D %s\n", dest->buf);
		  recursive_remove (dest->buf);
		}
	      break;

	    default:
	      if (verbose)
		printf ("-F %s\n", dest->buf);
	      maybe_xunlink (dest->buf);
	      break;
	    }
	}
    }

  closedir (dir);
}

static void
rsync (char *src, char *dest, int and_delete, int force_copies)
{
  r_setup (src, &spath);
  r_setup (dest, &dpath);

  rsync_1 (&spath, &dpath, and_delete, force_copies);
}



/* See if we can detect what the user needs to do to get unshare
   support working for us.  */
void
check_for_unshare_hints (void)
{
  FILE *f;
  int i;

  /* Default Debian Linux disables user namespaces, but allows a way
     to enable them.  */
  f = fopen ("/proc/sys/kernel/unprivileged_userns_clone", "r");
  if (f != NULL)
    {
      i = 99; /* Sentinel.  */
      fscanf (f, "%d", &i);
      if (i == 0)
	{
	  printf ("To enable test-container, please run this as root:\n");
	  printf ("  echo 1 > /proc/sys/kernel/unprivileged_userns_clone\n");
	}
      fclose (f);
      return;
    }

  /* ALT Linux has an alternate way of doing the same.  */
  f = fopen ("/proc/sys/kernel/userns_restrict", "r");
  if (f != NULL)
    {
      i = 99; /* Sentinel.  */
      fscanf (f, "%d", &i);
      if (i == 1)
	{
	  printf ("To enable test-container, please run this as root:\n");
	  printf ("  echo 0 > /proc/sys/kernel/userns_restrict\n");
	}
      fclose (f);
      return;
    }
}

static void
run_ldconfig (void *x __attribute__((unused)))
{
  char *prog = xasprintf ("%s/ldconfig", support_install_rootsbindir);
  char *args[] = { prog, NULL };

  execv (args[0], args);
  FAIL_EXIT1 ("execv: %m");
}

int
main (int argc, char **argv)
{
  pid_t child;
  char *pristine_root_path;
  char *new_root_path;
  char *new_cwd_path;
  char *new_objdir_path;
  char *new_srcdir_path;
  char **new_child_proc;
  char *new_child_exec;
  char *command_root;
  char *command_base;
  char *command_basename;
  char *so_base;
  int do_postclean = 0;
  bool do_ldconfig = false;
  char *change_cwd = NULL;

  int pipes[2];
  char pid_buf[20];

  uid_t original_uid;
  gid_t original_gid;
  /* If set, the test runs as root instead of the user running the testsuite.  */
  int be_su = 0;
  int UMAP;
  int GMAP;
  /* Used for "%lld %lld 1" so need not be large.  */
  char tmp[100];
  struct stat st;
  int lock_fd;

  setbuf (stdout, NULL);

  /* The command line we're expecting looks like this:
     env <set some vars> ld.so <library path> test-binary

     We need to peel off any "env" or "ld.so" portion of the command
     line, and keep track of which env vars we should preserve and
     which we drop.  */

  if (argc < 2)
    {
      fprintf (stderr, "Usage: test-container <program to run> <args...>\n");
      exit (1);
    }

  if (strcmp (argv[1], "-v") == 0)
    {
      verbose = 1;
      ++argv;
      --argc;
    }

  if (strcmp (argv[1], "env") == 0)
    {
      ++argv;
      --argc;
      while (is_env_setting (argv[1]))
	{
	  /* If there are variables we do NOT want to propogate, this
	     is where the test for them goes.  */
	    {
	      /* Need to keep these.  Note that putenv stores a
	         pointer to our argv.  */
	      putenv (argv[1]);
	    }
	  ++argv;
	  --argc;
	}
    }

  if (strcmp (argv[1], support_objdir_elf_ldso) == 0)
    {
      ++argv;
      --argc;
      while (argv[1][0] == '-')
	{
	  if (strcmp (argv[1], "--library-path") == 0)
	    {
	      ++argv;
	      --argc;
	    }
	  ++argv;
	  --argc;
	}
    }

  pristine_root_path = xstrdup (concat (support_objdir_root,
				       "/testroot.pristine", NULL));
  new_root_path = xstrdup (concat (support_objdir_root,
				  "/testroot.root", NULL));
  new_cwd_path = get_current_dir_name ();
  new_child_proc = argv + 1;
  new_child_exec = argv[1];

  lock_fd = open (concat (pristine_root_path, "/lock.fd", NULL),
		 O_CREAT | O_TRUNC | O_RDWR, 0666);
  if (lock_fd < 0)
    FAIL_EXIT1 ("Cannot create testroot lock.\n");

  while (flock (lock_fd, LOCK_EX) != 0)
    {
      if (errno != EINTR)
	FAIL_EXIT1 ("Cannot lock testroot.\n");
    }

  xmkdirp (new_root_path, 0755);

  /* We look for extra setup info in a subdir in the same spot as the
     test, with the same name but a ".root" extension.  This is that
     directory.  We try to look in the source tree if the path we're
     given refers to the build tree, but we rely on the path to be
     absolute.  This is what the glibc makefiles do.  */
  command_root = concat (argv[1], ".root", NULL);
  if (strncmp (command_root, support_objdir_root,
	       strlen (support_objdir_root)) == 0
      && command_root[strlen (support_objdir_root)] == '/')
    command_root = concat (support_srcdir_root,
			   argv[1] + strlen (support_objdir_root),
			   ".root", NULL);
  command_root = xstrdup (command_root);

  /* This cuts off the ".root" we appended above.  */
  command_base = xstrdup (command_root);
  command_base[strlen (command_base) - 5] = 0;

  /* This is the basename of the test we're running.  */
  command_basename = strrchr (command_base, '/');
  if (command_basename == NULL)
    command_basename = command_base;
  else
    ++command_basename;

  /* Shared object base directory.  */
  so_base = xstrdup (argv[1]);
  if (strrchr (so_base, '/') != NULL)
    strrchr (so_base, '/')[1] = 0;

  if (file_exists (concat (command_root, "/postclean.req", NULL)))
    do_postclean = 1;

  if (file_exists (concat (command_root, "/ldconfig.run", NULL)))
    do_ldconfig = true;

  rsync (pristine_root_path, new_root_path,
	 file_exists (concat (command_root, "/preclean.req", NULL)), 0);

  if (stat (command_root, &st) >= 0
      && S_ISDIR (st.st_mode))
    rsync (command_root, new_root_path, 0, 1);

  new_objdir_path = xstrdup (concat (new_root_path,
				    support_objdir_root, NULL));
  new_srcdir_path = xstrdup (concat (new_root_path,
				    support_srcdir_root, NULL));

  /* new_cwd_path starts with '/' so no "/" needed between the two.  */
  xmkdirp (concat (new_root_path, new_cwd_path, NULL), 0755);
  xmkdirp (new_srcdir_path, 0755);
  xmkdirp (new_objdir_path, 0755);

  original_uid = getuid ();
  original_gid = getgid ();

  /* Handle the cp/mv/rm "script" here.  */
  {
    char *the_line = NULL;
    size_t line_len = 0;
    char *fname = concat (command_root, "/",
			  command_basename, ".script", NULL);
    char *the_words[3];
    FILE *f = fopen (fname, "r");

    if (verbose && f)
      fprintf (stderr, "running %s\n", fname);

    if (f == NULL)
      {
	/* Try foo.script instead of foo.root/foo.script, as a shortcut.  */
	fname = concat (command_base, ".script", NULL);
	f = fopen (fname, "r");
	if (verbose && f)
	  fprintf (stderr, "running %s\n", fname);
      }

    /* Note that we do NOT look for a Makefile-generated foo.script in
       the build directory.  If that is ever needed, this is the place
       to add it.  */

    /* This is where we "interpret" the mini-script which is <test>.script.  */
    if (f != NULL)
      {
	while (getline (&the_line, &line_len, f) > 0)
	  {
	    int nt = tokenize (the_line, the_words, 3);
	    int i;

	    /* Expand variables.  */
	    for (i = 1; i < nt; ++i)
	      {
		if (memcmp (the_words[i], "$B/", 3) == 0)
		  the_words[i] = concat (support_objdir_root,
					 the_words[i] + 2, NULL);
		else if (memcmp (the_words[i], "$S/", 3) == 0)
		  the_words[i] = concat (support_srcdir_root,
					 the_words[i] + 2, NULL);
		else if (memcmp (the_words[i], "$I/", 3) == 0)
		  the_words[i] = concat (new_root_path,
					 support_install_prefix,
					 the_words[i] + 2, NULL);
		else if (memcmp (the_words[i], "$L/", 3) == 0)
		  the_words[i] = concat (new_root_path,
					 support_libdir_prefix,
					 the_words[i] + 2, NULL);
		else if (memcmp (the_words[i], "$complocaledir/", 15) == 0)
		  the_words[i] = concat (new_root_path,
					 support_complocaledir_prefix,
					 the_words[i] + 14, NULL);
		/* "exec" and "cwd" use inside-root paths.  */
		else if (strcmp (the_words[0], "exec") != 0
			 && strcmp (the_words[0], "cwd") != 0
			 && the_words[i][0] == '/')
		  the_words[i] = concat (new_root_path,
					 the_words[i], NULL);
	      }

	    if (nt == 3 && the_words[2][strlen (the_words[2]) - 1] == '/')
	      {
		char *r = strrchr (the_words[1], '/');
		if (r)
		  the_words[2] = concat (the_words[2], r + 1, NULL);
		else
		  the_words[2] = concat (the_words[2], the_words[1], NULL);
	      }

	    /* Run the following commands in the_words[0] with NT number of
	       arguments (including the command).  */

	    if (nt == 2 && strcmp (the_words[0], "so") == 0)
	      {
		the_words[2] = concat (new_root_path, support_libdir_prefix,
				       "/", the_words[1], NULL);
		the_words[1] = concat (so_base, the_words[1], NULL);
		copy_one_file (the_words[1], the_words[2]);
	      }
	    else if (nt == 3 && strcmp (the_words[0], "cp") == 0)
	      {
		copy_one_file (the_words[1], the_words[2]);
	      }
	    else if (nt == 3 && strcmp (the_words[0], "mv") == 0)
	      {
		if (rename (the_words[1], the_words[2]) < 0)
		  FAIL_EXIT1 ("rename %s -> %s: %s", the_words[1],
			      the_words[2], strerror (errno));
	      }
	    else if (nt == 3 && strcmp (the_words[0], "chmod") == 0)
	      {
		long int m;
		errno = 0;
		m = strtol (the_words[1], NULL, 0);
		TEST_COMPARE (errno, 0);
		if (chmod (the_words[2], m) < 0)
		    FAIL_EXIT1 ("chmod %s: %s\n",
				the_words[2], strerror (errno));

	      }
	    else if (nt == 2 && strcmp (the_words[0], "rm") == 0)
	      {
		maybe_xunlink (the_words[1]);
	      }
	    else if (nt >= 2 && strcmp (the_words[0], "exec") == 0)
	      {
		/* The first argument is the desired location and name
		   of the test binary as we wish to exec it; we will
		   copy the binary there.  The second (optional)
		   argument is the value to pass as argv[0], it
		   defaults to the same as the first argument.  */
		char *new_exec_path = the_words[1];

		/* If the new exec path ends with a slash, that's the
		 * directory, and use the old test base name.  */
		if (new_exec_path [strlen(new_exec_path) - 1] == '/')
		    new_exec_path = concat (new_exec_path,
					    basename (new_child_proc[0]),
					    NULL);


		/* new_child_proc is in the build tree, so has the
		   same path inside the chroot as outside.  The new
		   exec path is, by definition, relative to the
		   chroot.  */
		copy_one_file (new_child_proc[0],  concat (new_root_path,
							   new_exec_path,
							   NULL));

		new_child_exec =  xstrdup (new_exec_path);
		if (the_words[2])
		  new_child_proc[0] = xstrdup (the_words[2]);
		else
		  new_child_proc[0] = new_child_exec;
	      }
	    else if (nt == 2 && strcmp (the_words[0], "cwd") == 0)
	      {
		change_cwd = xstrdup (the_words[1]);
	      }
	    else if (nt == 1 && strcmp (the_words[0], "su") == 0)
	      {
		be_su = 1;
	      }
	    else if (nt == 3 && strcmp (the_words[0], "mkdirp") == 0)
	      {
		long int m;
		errno = 0;
		m = strtol (the_words[1], NULL, 0);
		TEST_COMPARE (errno, 0);
		xmkdirp (the_words[2], m);
	      }
	    else if (nt > 0 && the_words[0][0] != '#')
	      {
		fprintf (stderr, "\033[31minvalid [%s]\033[0m\n", the_words[0]);
		exit (1);
	      }
	  }
	fclose (f);
      }
  }

  if (do_postclean)
    {
      pid_t pc_pid = fork ();

      if (pc_pid < 0)
	{
	  FAIL_EXIT1 ("Can't fork for post-clean");
	}
      else if (pc_pid > 0)
	{
	  /* Parent.  */
	  int status;
	  waitpid (pc_pid, &status, 0);

	  /* Child has exited, we can post-clean the test root.  */
	  printf("running post-clean rsync\n");
	  rsync (pristine_root_path, new_root_path, 1, 0);

	  if (WIFEXITED (status))
	    exit (WEXITSTATUS (status));

	  if (WIFSIGNALED (status))
	    {
	      printf ("%%SIGNALLED%%\n");
	      exit (77);
	    }

	  printf ("%%EXITERROR%%\n");
	  exit (78);
	}

      /* Child continues.  */
    }

  /* This is the last point in the program where we're still in the
     "normal" namespace.  */

#ifdef CLONE_NEWNS
  /* The unshare here gives us our own spaces and capabilities.  */
  if (unshare (CLONE_NEWUSER | CLONE_NEWPID | CLONE_NEWNS) < 0)
    {
      /* Older kernels may not support all the options, or security
	 policy may block this call.  */
      if (errno == EINVAL || errno == EPERM)
	{
	  int saved_errno = errno;
	  if (errno == EPERM)
	    check_for_unshare_hints ();
	  FAIL_UNSUPPORTED ("unable to unshare user/fs: %s", strerror (saved_errno));
	}
      else
	FAIL_EXIT1 ("unable to unshare user/fs: %s", strerror (errno));
    }
#else
  /* Some targets may not support unshare at all.  */
  FAIL_UNSUPPORTED ("unshare support missing");
#endif

  /* Some systems, by default, all mounts leak out of the namespace.  */
  if (mount ("none", "/", NULL, MS_REC | MS_PRIVATE, NULL) != 0)
    FAIL_EXIT1 ("could not create a private mount namespace\n");

  trymount (support_srcdir_root, new_srcdir_path);
  trymount (support_objdir_root, new_objdir_path);

  xmkdirp (concat (new_root_path, "/dev", NULL), 0755);
  devmount (new_root_path, "null");
  devmount (new_root_path, "zero");
  devmount (new_root_path, "urandom");

  /* We're done with the "old" root, switch to the new one.  */
  if (chroot (new_root_path) < 0)
    FAIL_EXIT1 ("Can't chroot to %s - ", new_root_path);

  if (chdir (new_cwd_path) < 0)
    FAIL_EXIT1 ("Can't cd to new %s - ", new_cwd_path);

  /* This is to pass the "outside" PID to the child, which will be PID
     1.  */
  if (pipe2 (pipes, O_CLOEXEC) < 0)
    FAIL_EXIT1 ("Can't create pid pipe");

  /* To complete the containerization, we need to fork () at least
     once.  We can't exec, nor can we somehow link the new child to
     our parent.  So we run the child and propogate it's exit status
     up.  */
  child = fork ();
  if (child < 0)
    FAIL_EXIT1 ("Unable to fork");
  else if (child > 0)
    {
      /* Parent.  */
      int status;

      /* Send the child's "outside" pid to it.  */
      write (pipes[1], &child, sizeof(child));
      close (pipes[0]);
      close (pipes[1]);

      waitpid (child, &status, 0);

      if (WIFEXITED (status))
	exit (WEXITSTATUS (status));

      if (WIFSIGNALED (status))
	{
	  printf ("%%SIGNALLED%%\n");
	  exit (77);
	}

      printf ("%%EXITERROR%%\n");
      exit (78);
    }

  /* The rest is the child process, which is now PID 1 and "in" the
     new root.  */

  if (do_ldconfig)
    {
      struct support_capture_subprocess result =
        support_capture_subprocess (run_ldconfig, NULL);
      support_capture_subprocess_check (&result, "execv", 0, sc_allow_none);
    }

  /* Get our "outside" pid from our parent.  We use this to help with
     debugging from outside the container.  */
  read (pipes[0], &child, sizeof(child));
  close (pipes[0]);
  close (pipes[1]);
  sprintf (pid_buf, "%lu", (long unsigned)child);
  setenv ("PID_OUTSIDE_CONTAINER", pid_buf, 0);

  maybe_xmkdir ("/tmp", 0755);

  /* Now that we're pid 1 (effectively "root") we can mount /proc  */
  maybe_xmkdir ("/proc", 0777);
  if (mount ("proc", "/proc", "proc", 0, NULL) < 0)
    FAIL_EXIT1 ("Unable to mount /proc: ");

  /* We map our original UID to the same UID in the container so we
     can own our own files normally.  */
  UMAP = open ("/proc/self/uid_map", O_WRONLY);
  if (UMAP < 0)
    FAIL_EXIT1 ("can't write to /proc/self/uid_map\n");

  sprintf (tmp, "%lld %lld 1\n",
	   (long long) (be_su ? 0 : original_uid), (long long) original_uid);
  write (UMAP, tmp, strlen (tmp));
  xclose (UMAP);

  /* We must disable setgroups () before we can map our groups, else we
     get EPERM.  */
  GMAP = open ("/proc/self/setgroups", O_WRONLY);
  if (GMAP >= 0)
    {
      /* We support kernels old enough to not have this.  */
      write (GMAP, "deny\n", 5);
      xclose (GMAP);
    }

  /* We map our original GID to the same GID in the container so we
     can own our own files normally.  */
  GMAP = open ("/proc/self/gid_map", O_WRONLY);
  if (GMAP < 0)
    FAIL_EXIT1 ("can't write to /proc/self/gid_map\n");

  sprintf (tmp, "%lld %lld 1\n",
	   (long long) (be_su ? 0 : original_gid), (long long) original_gid);
  write (GMAP, tmp, strlen (tmp));
  xclose (GMAP);

  if (change_cwd)
    {
      if (chdir (change_cwd) < 0)
	FAIL_EXIT1 ("Can't cd to %s inside container - ", change_cwd);
    }

  /* Now run the child.  */
  execvp (new_child_exec, new_child_proc);

  /* Or don't run the child?  */
  FAIL_EXIT1 ("Unable to exec %s: %s\n", new_child_exec, strerror (errno));

  /* Because gcc won't know error () never returns...  */
  exit (EXIT_UNSUPPORTED);
}
