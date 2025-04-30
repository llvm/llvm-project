/* Check handling of missing end-of-line at end of /etc/aliases (bug 24059).
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#include <aliases.h>
#include <gnu/lib-names.h>
#include <nss.h>
#include <stddef.h>
#include <support/check.h>
#include <support/namespace.h>
#include <support/test-driver.h>
#include <support/xdlfcn.h>
#include <support/xunistd.h>

static void
in_chroot (void *closure)
{
  struct support_chroot *chroot_env = closure;
  xchroot (chroot_env->path_chroot);

  struct aliasent *e = getaliasbyname ("user1");
  TEST_VERIFY_EXIT (e != NULL);
  TEST_COMPARE_STRING (e->alias_name, "user1");
  TEST_COMPARE (e->alias_members_len, 1);
  TEST_VERIFY_EXIT (e->alias_members != NULL);
  TEST_COMPARE_STRING (e->alias_members[0], "alias1");
  TEST_VERIFY (e->alias_local);
}

static int
do_test (void)
{
  /* Make sure we don't try to load the module in the chroot.  */
  xdlopen (LIBNSS_FILES_SO, RTLD_NOW);

  __nss_configure_lookup ("aliases", "files");

  support_become_root ();
  if (!support_can_chroot ())
    return EXIT_UNSUPPORTED;

  struct support_chroot *chroot_env = support_chroot_create
    ((struct support_chroot_configuration)
     {
       .aliases = "user1: alias1,\n"
        " "              /* Continuation line, but no \n.  */
     });

  support_isolate_in_subprocess (in_chroot, chroot_env);

  support_chroot_free (chroot_env);
  return 0;
}

#include <support/test-driver.c>
