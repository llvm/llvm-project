/* Enter a network namespace.
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

#include <net/if.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <support/check.h>
#include <support/xsocket.h>
#include <support/xunistd.h>
#include <sys/ioctl.h>
#include <unistd.h>

static bool in_uts_namespace;

bool
support_enter_network_namespace (void)
{
#ifdef CLONE_NEWUTS
  if (unshare (CLONE_NEWUTS) == 0)
    in_uts_namespace = true;
  else
    printf ("warning: unshare (CLONE_NEWUTS) failed: %m\n");
#endif

#ifdef CLONE_NEWNET
  if (unshare (CLONE_NEWNET) == 0)
    {
      /* Bring up the loopback interface.  */
      int fd = xsocket (AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
      struct ifreq req;
      strcpy (req.ifr_name, "lo");
      TEST_VERIFY_EXIT (ioctl (fd, SIOCGIFFLAGS, &req) == 0);
      bool already_up = req.ifr_flags & IFF_UP;
      if (already_up)
        /* This means that we likely have not achieved isolation from
           the parent namespace.  */
        printf ("warning: loopback interface already exists"
                " in new network namespace\n");
      else
        {
          req.ifr_flags |= IFF_UP | IFF_RUNNING;
          TEST_VERIFY_EXIT (ioctl (fd, SIOCSIFFLAGS, &req) == 0);
        }
      xclose (fd);

      return !already_up;
    }
#endif
  printf ("warning: could not enter network namespace\n");
  return false;
}

bool
support_in_uts_namespace (void)
{
  return in_uts_namespace;
}
