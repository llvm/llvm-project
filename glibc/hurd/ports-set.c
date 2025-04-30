/* Copyright (C) 1994-2021 Free Software Foundation, Inc.
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

static error_t
setbootstrap (mach_port_t newport)
{
  return __task_set_special_port (__mach_task_self (),
				  TASK_BOOTSTRAP_PORT,
				  newport);
}

extern error_t _hurd_setauth (auth_t);
extern error_t _hurd_setproc (process_t);
extern error_t _hurd_setcttyid (mach_port_t);

error_t (*_hurd_ports_setters[INIT_PORT_MAX]) (mach_port_t newport) =
  {
    [INIT_PORT_BOOTSTRAP] = setbootstrap,
    [INIT_PORT_AUTH] = _hurd_setauth,
    [INIT_PORT_PROC] = _hurd_setproc,
    [INIT_PORT_CTTYID] = _hurd_setcttyid,
  };


error_t
_hurd_ports_set (unsigned int which, mach_port_t newport)
{
  error_t err;
  if (which >= _hurd_nports)
    return EINVAL;
  if (err = __mach_port_mod_refs (__mach_task_self (), newport,
				  MACH_PORT_RIGHT_SEND, 1))
    return err;
  if (which >= INIT_PORT_MAX || _hurd_ports_setters[which] == NULL)
    {
      _hurd_port_set (&_hurd_ports[which], newport);
      return 0;
    }
  return (*_hurd_ports_setters[which]) (newport);
}
