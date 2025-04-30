/* Access to hardware i/o ports.  Hurd/x86 version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#include <sys/io.h>
#include <hurd.h>
#include <mach/i386/mach_i386.h>

int
ioperm (unsigned long int from, unsigned long int num, int turn_on)
{
#if ! HAVE_I386_IO_PERM_MODIFY
  return __hurd_fail (ENOSYS);
#else
  error_t err;
  device_t devmaster;

  /* With the device master port we get a capability that represents
     this range of io ports.  */
  err = __get_privileged_ports (NULL, &devmaster);
  if (! err)
    {
      io_perm_t perm;
      err = __i386_io_perm_create (devmaster, from, from + num - 1, &perm);
      __mach_port_deallocate (__mach_task_self (), devmaster);
      if (! err)
	{
	  /* Now we add or remove that set from our task's bitmap.  */
	  err = __i386_io_perm_modify (__mach_task_self (), perm, turn_on);
	  __mach_port_deallocate (__mach_task_self (), perm);
	}

      if (err == MIG_BAD_ID)	/* Old kernels don't have these RPCs.  */
	err = ENOSYS;
    }

  return err ? __hurd_fail (err) : 0;
#endif
}
