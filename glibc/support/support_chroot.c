/* Setup a chroot environment for use within tests.
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

#include <stdlib.h>
#include <support/check.h>
#include <support/namespace.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <support/test-driver.h>
#include <support/xunistd.h>

/* If CONTENTS is not NULL, write it to the file at DIRECTORY/RELPATH,
   and store the name in *ABSPATH.  If CONTENTS is NULL, store NULL in
   *ABSPATH.  */
static void
write_file (const char *directory, const char *relpath, const char *contents,
            char **abspath)
{
  if (contents != NULL)
    {
      *abspath = xasprintf ("%s/%s", directory, relpath);
      add_temp_file (*abspath);
      support_write_file_string (*abspath, contents);
    }
  else
    *abspath = NULL;
}

struct support_chroot *
support_chroot_create (struct support_chroot_configuration conf)
{
  struct support_chroot *chroot = xmalloc (sizeof (*chroot));
  chroot->path_chroot = support_create_temp_directory ("tst-resolv-res_init-");

  /* Create the /etc directory in the chroot environment.  */
  char *path_etc = xasprintf ("%s/etc", chroot->path_chroot);
  xmkdir (path_etc, 0777);
  add_temp_file (path_etc);

  write_file (path_etc, "resolv.conf", conf.resolv_conf,
              &chroot->path_resolv_conf);
  write_file (path_etc, "hosts", conf.hosts, &chroot->path_hosts);
  write_file (path_etc, "host.conf", conf.host_conf, &chroot->path_host_conf);
  write_file (path_etc, "aliases", conf.aliases, &chroot->path_aliases);

  free (path_etc);

  /* valgrind needs a temporary directory in the chroot.  */
  {
    char *path_tmp = xasprintf ("%s/tmp", chroot->path_chroot);
    xmkdir (path_tmp, 0777);
    add_temp_file (path_tmp);
    free (path_tmp);
  }

  return chroot;
}

void
support_chroot_free (struct support_chroot *chroot)
{
  free (chroot->path_chroot);
  free (chroot->path_resolv_conf);
  free (chroot->path_hosts);
  free (chroot->path_host_conf);
  free (chroot->path_aliases);
  free (chroot);
}
