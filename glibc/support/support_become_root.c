/* Acquire root privileges.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <support/namespace.h>

#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <support/check.h>
#include <support/xunistd.h>
#include <unistd.h>

#ifdef CLONE_NEWUSER
/* The necessary steps to allow file creation in user namespaces.  */
static void
setup_uid_gid_mapping (uid_t original_uid, gid_t original_gid)
{
  int fd = open64 ("/proc/self/uid_map", O_WRONLY);
  if (fd < 0)
    {
      printf ("warning: could not open /proc/self/uid_map: %m\n"
              "warning: file creation may not be possible\n");
      return;
    }

  /* We map our original UID to the same UID in the container so we
     own our own files normally.  Without that, file creation could
     fail with EOVERFLOW (sic!).  */
  char buf[100];
  int ret = snprintf (buf, sizeof (buf), "%llu %llu 1\n",
                      (unsigned long long) original_uid,
                      (unsigned long long) original_uid);
  TEST_VERIFY_EXIT (ret < sizeof (buf));
  xwrite (fd, buf, ret);
  xclose (fd);

  /* Linux 3.19 introduced the setgroups file.  We need write "deny" to this
     file otherwise writing to gid_map will fail with EPERM.  */
  fd = open64 ("/proc/self/setgroups", O_WRONLY, 0);
  if (fd < 0)
    {
      if (errno != ENOENT)
        FAIL_EXIT1 ("open64 (\"/proc/self/setgroups\", 0x%x, 0%o): %m",
                    O_WRONLY, 0);
      /* This kernel doesn't expose the setgroups file so simply move on.  */
    }
  else
    {
      xwrite (fd, "deny\n", strlen ("deny\n"));
      xclose (fd);
    }

  /* Now map our own GID, like we did for the user ID.  */
  fd = xopen ("/proc/self/gid_map", O_WRONLY, 0);
  ret = snprintf (buf, sizeof (buf), "%llu %llu 1\n",
                  (unsigned long long) original_gid,
                  (unsigned long long) original_gid);
  TEST_VERIFY_EXIT (ret < sizeof (buf));
  xwrite (fd, buf, ret);
  xclose (fd);
}
#endif /* CLONE_NEWUSER */

bool
support_become_root (void)
{
#ifdef CLONE_NEWUSER
  uid_t original_uid = getuid ();
  gid_t original_gid = getgid ();

  if (unshare (CLONE_NEWUSER | CLONE_NEWNS) == 0)
    {
      setup_uid_gid_mapping (original_uid, original_gid);
      /* Even if we do not have UID zero, we have extended privileges at
         this point.  */
      return true;
    }
#endif
  if (setuid (0) != 0)
    {
      printf ("warning: could not become root outside namespace (%m)\n");
      return false;
    }
  return true;
}
