/* Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>


static long int linux_sysconf (int name);

/* Get the value of the system variable NAME.  */
long int
__sysconf (int name)
{
  unsigned ctr;

  /* Unfortunately, the registers that contain the actual cache info
     (CCSIDR_EL1, CLIDR_EL1, and CSSELR_EL1) are protected by the Linux
     kernel (though they need not have been).  However, CTR_EL0 contains
     the *minimum* linesize in the entire cache hierarchy, and is
     accessible to userland, for use in __aarch64_sync_cache_range,
     and it is a reasonable assumption that the L1 cache will have that
     minimum line size.  */
  switch (name)
    {
    case _SC_LEVEL1_ICACHE_LINESIZE:
      asm("mrs\t%0, ctr_el0" : "=r"(ctr));
      return 4 << (ctr & 0xf);
    case _SC_LEVEL1_DCACHE_LINESIZE:
      asm("mrs\t%0, ctr_el0" : "=r"(ctr));
      return 4 << ((ctr >> 16) & 0xf);
    }

  return linux_sysconf (name);
}

/* Now the generic Linux version.  */
#undef __sysconf
#define __sysconf static linux_sysconf
#include <sysdeps/unix/sysv/linux/sysconf.c>
