/* Check if realpath does not consume extra stack space based on symlink
   existance in the path (BZ #26341)
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <unistd.h>

#define __sysconf sysconf
#include <eloop-threshold.h>
#include <support/check.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <support/xunistd.h>
#include <support/xthread.h>

static char *filename;
static size_t filenamelen;
static char *linkname;

#ifndef PATH_MAX
# define PATH_MAX 1024
#endif

static void
create_link (void)
{
  int fd = create_temp_file ("tst-canon-bz26341", &filename);
  TEST_VERIFY_EXIT (fd != -1);
  xclose (fd);

  /* Create MAXLINKS symbolic links to the temporary filename.
     On exit, linkname has the last link created.  */
  char *prevlink = filename;
  int maxlinks = __eloop_threshold ();
  for (int i = 0; i < maxlinks; i++)
    {
      linkname = xasprintf ("%s%d", filename, i);
      xsymlink (prevlink, linkname);
      add_temp_file (linkname);
      prevlink = linkname;
    }

  filenamelen = strlen (filename);
}

static void *
do_realpath (void *arg)
{
  /* Old implementation of realpath allocates a PATH_MAX using alloca
     for each symlink in the path, leading to MAXSYMLINKS times PATH_MAX
     maximum stack usage.
     This stack allocations tries fill the thread allocated stack minus
     the resolved path (plus some slack), the realpath (plus some
     slack), and the system call usage (plus some slack).
     If realpath uses more than 2 * PATH_MAX plus some slack it will trigger
     a stackoverflow.  */

  const size_t syscall_usage = 1 * PATH_MAX + 1024;
  const size_t realpath_usage = 2 * PATH_MAX + 1024;
  const size_t thread_usage = 1 * PATH_MAX + 1024;
  size_t stack_size = support_small_thread_stack_size ()
		      - syscall_usage - realpath_usage - thread_usage;
  char stack[stack_size];
  char *resolved = stack + stack_size - thread_usage + 1024;

  char *p = realpath (linkname, resolved);
  TEST_VERIFY (p != NULL);
  TEST_COMPARE_BLOB (resolved, filenamelen, filename, filenamelen);

  return NULL;
}

static int
do_test (void)
{
  create_link ();

  pthread_t th = xpthread_create (support_small_stack_thread_attribute (),
				  do_realpath, NULL);
  xpthread_join (th);

  return 0;
}

#include <support/test-driver.c>
