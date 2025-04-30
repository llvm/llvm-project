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

#include <unistd.h>
#include <hurd.h>
#include <fcntl.h>
#include <hurd/port.h>

/* Change the current directory to FILE_NAME.  */
int
__chdir (const char *file_name)
{
  return _hurd_change_directory_port_from_name (&_hurd_ports[INIT_PORT_CWDIR],
						file_name);
}

weak_alias (__chdir, chdir)
