/* Basic test for gethostid.
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

#include <gnu/lib-names.h>
#include <nss.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/namespace.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <support/xdlfcn.h>
#include <support/xstdio.h>
#include <support/xunistd.h>
#include <unistd.h>

/* Initial test is run outside a chroot, to increase the likelihood of
   success.  */
static void
outside_chroot (void *closure)
{
  long id = gethostid ();
  printf ("info: host ID outside chroot: 0x%lx\n", id);
}

/* The same, but this time perform a chroot operation.  */
static void
in_chroot (void *closure)
{
  const char *chroot_path = closure;
  xchroot (chroot_path);
  long id = gethostid ();
  printf ("info: host ID in chroot: 0x%lx\n", id);
}

static int
do_test (void)
{
  support_isolate_in_subprocess (outside_chroot, NULL);

  /* Now run the test inside a chroot.  */
  support_become_root ();
  if (!support_can_chroot ())
    /* Cannot perform further tests.  */
    return 0;

  /* Only use nss_files.  */
  __nss_configure_lookup ("hosts", "files");

  /* Load the DSO outside of the chroot.  */
  xdlopen (LIBNSS_FILES_SO, RTLD_LAZY);

  char *chroot_dir = support_create_temp_directory ("tst-gethostid-");
  support_isolate_in_subprocess (in_chroot, chroot_dir);

  /* Tests with /etc/hosts in the chroot.  */
  {
    char *path = xasprintf ("%s/etc", chroot_dir);
    add_temp_file (path);
    xmkdir (path, 0777);
    free (path);
    path = xasprintf ("%s/etc/hosts", chroot_dir);
    add_temp_file (path);

    FILE *fp = xfopen (path, "w");
    xfclose (fp);
    printf ("info: chroot test with an empty /etc/hosts file\n");
    support_isolate_in_subprocess (in_chroot, chroot_dir);

    char hostname[1024];
    int ret = gethostname (hostname, sizeof (hostname));
    if (ret < 0)
      printf ("warning: invalid result from gethostname: %d\n", ret);
    else if (strlen (hostname) == 0)
      puts ("warning: gethostname returned empty string");
    else
      {
        printf ("info: chroot test with IPv6 address in /etc/hosts for: %s\n",
                hostname);
        fp = xfopen (path, "w");
        /* Use an IPv6 address to induce another lookup failure.  */
        fprintf (fp, "2001:db8::1 %s\n", hostname);
        xfclose (fp);
        support_isolate_in_subprocess (in_chroot, chroot_dir);
      }
    free (path);
  }
  free (chroot_dir);

  return 0;
}

#include <support/test-driver.c>
