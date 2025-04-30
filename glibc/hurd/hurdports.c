/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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
#include <hurd/port.h>


static inline mach_port_t
get (const int idx)
{
  mach_port_t result;
  error_t err = _hurd_ports_get (idx, &result);

  if (err)
    return __hurd_fail (err), MACH_PORT_NULL;
  return result;
}
#define	GET(type, what, idx) \
  type get##what (void) { return get (INIT_PORT_##idx); }

static inline int
set (const int idx, mach_port_t new)
{
  error_t err = _hurd_ports_set (idx, new);
  return err ? __hurd_fail (err) : 0;
}
#define SET(type, what, idx) \
  int set##what (type new) { return set (INIT_PORT_##idx, new); }

#define	GETSET(type, what, idx) \
  GET (type, what, idx) SET (type, what, idx)

GETSET (process_t, proc, PROC)
GETSET (mach_port_t, cttyid, CTTYID)
GETSET (file_t, cwdir, CWDIR)
GETSET (file_t, crdir, CRDIR)
GETSET (auth_t, auth, AUTH)
