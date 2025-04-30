/* Compat socket constants used in 64-bit compat code.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#ifndef _SOCKET_CONSTANTS_TIME64_H
#define _SOCKET_CONSTANTS_TIME64_H

/* The compat code requires the SO_* constants used for both 32 and 64-bit
   time_t, however they were only added on v5.1 kernel.  */

#define COMPAT_SO_RCVTIMEO_OLD 8192
#define COMPAT_SO_SNDTIMEO_OLD 16384
#define COMPAT_SO_RCVTIMEO_NEW 68
#define COMPAT_SO_SNDTIMEO_NEW 69

#define COMPAT_SO_TIMESTAMP_OLD 0x001d
#define COMPAT_SO_TIMESTAMPNS_OLD 0x0021
#define COMPAT_SO_TIMESTAMP_NEW 0x0046
#define COMPAT_SO_TIMESTAMPNS_NEW 0x0042

#endif
