/* Device numbers of devices used in the implementation.  Linux version.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#ifndef _DEVICE_NRS_H
#define _DEVICE_NRS_H	1

#include <sys/sysmacros.h>

/* /dev/null is (1,3).  */
#define DEV_NULL_MAJOR	1
#define DEV_NULL_MINOR	3

/* /dev/full is (1,7).  */
#define DEV_FULL_MAJOR	1
#define DEV_FULL_MINOR	7

/* Pseudo tty slaves.  For Linux we use the Unix98 ttys.  We could
   also include the old BSD-style tty buts they should not be used and
   the extra test would only slow down correctly set up systems.  If a
   system still uses those device the slower tests performed (using
   isatty) will catch it.  */
#define DEV_TTY_LOW_MAJOR	136
#define DEV_TTY_HIGH_MAJOR	143

/* Test whether given device is a tty.  */
#define DEV_TTY_P(statp) \
  ({ int __dev_major = __gnu_dev_major ((statp)->st_rdev);		      \
     __dev_major >= DEV_TTY_LOW_MAJOR && __dev_major <= DEV_TTY_HIGH_MAJOR; })

#endif	/* device-nrs.h */
