/* Copyright (C) 1992-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <unistd.h>
#include <hurd.h>
#include <hurd/paths.h>
#include <hurd/startup.h>
#include <sys/reboot.h>

/* Reboot the system.  */
int
reboot (int howto)
{
  error_t err;
  startup_t init;
  mach_port_t hostpriv;

  err = __get_privileged_ports (&hostpriv, NULL);
  if (err)
    return __hurd_fail (EPERM);

  init = __file_name_lookup (_SERVERS_STARTUP, 0, 0);
  if (init != MACH_PORT_NULL)
    {
      err = __startup_reboot (init, hostpriv, howto);
      __mach_port_deallocate (__mach_task_self (), init);
    }

  __mach_port_deallocate (__mach_task_self (), hostpriv);

  if (err)
    return __hurd_fail (err);

  return 0;
}
