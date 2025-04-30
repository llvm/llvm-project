/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
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

#include <hurd.h>

/* The program might set these if it is the initial task
   bootstrapped by the microkernel.  */

mach_port_t _hurd_host_priv, _hurd_device_master;


kern_return_t
__get_privileged_ports (mach_port_t *host_priv_ptr,
			device_t *device_master_ptr)
{
  if ((host_priv_ptr && _hurd_host_priv == MACH_PORT_NULL)
      || (device_master_ptr && _hurd_device_master == MACH_PORT_NULL))
    {
      error_t err;

      if (_hurd_ports)
	/* We have gotten some initial ports, so perhaps
	   we have a proc server to talk to.  */
	err = __USEPORT (PROC, __proc_getprivports (port,
						    &_hurd_host_priv,
						    &_hurd_device_master));
      else
	return MACH_SEND_INVALID_DEST;

      if (err)
	return err;
    }

  if (host_priv_ptr)
    {
      error_t err = _hurd_host_priv == MACH_PORT_NULL ? 0
	: __mach_port_mod_refs (mach_task_self (),
				_hurd_host_priv, MACH_PORT_RIGHT_SEND, +1);
      if (err)
	return err;
      *host_priv_ptr = _hurd_host_priv;
    }

  if (device_master_ptr)
    {
      error_t err = _hurd_device_master == MACH_PORT_NULL ? 0
	: __mach_port_mod_refs (mach_task_self (),
				_hurd_device_master, MACH_PORT_RIGHT_SEND, +1);
      if (err)
	return err;
      *device_master_ptr = _hurd_device_master;
    }

  return KERN_SUCCESS;
}
weak_alias (__get_privileged_ports, get_privileged_ports)
