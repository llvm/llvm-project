/* Hurd-specific socket functions
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#ifndef _HURD_HURDSOCKET_H
#define _HURD_HURDSOCKET_H

#include <string.h>

/* Returns a duplicate of ADDR->sun_path with LEN limitation.  This
   should to be used whenever reading a unix socket address, to cope with
   sun_path possibly not including a trailing \0.  */
#define _hurd_sun_path_dupa(addr, len) \
  strndupa ((addr)->sun_path, (len) - offsetof (struct sockaddr_un, sun_path))

#endif /* hurdsocket.h */
