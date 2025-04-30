/* Get file-specific information about a file.  Linux version.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>


static long int linux_sysconf (int name);
extern long int __cache_sysconf (int) attribute_hidden;


/* Get the value of the system variable NAME.  */
long int
__sysconf (int name)
{
  if (name >= _SC_LEVEL1_ICACHE_SIZE && name <= _SC_LEVEL4_CACHE_LINESIZE)
    return __cache_sysconf (name);

  return linux_sysconf (name);
}

/* Now the generic Linux version.  */
#undef __sysconf
#define __sysconf static linux_sysconf
#include "../sysconf.c"
