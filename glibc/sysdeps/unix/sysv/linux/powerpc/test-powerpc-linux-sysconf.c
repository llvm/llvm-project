/* Check use of sysconf() for cache geometries.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

/* Test use of sysconf() to get cache sizes, cache set associativity
   and cache line sizes.  */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <support/test-driver.h>

#define call_str(f, name) f(name, #name)

long
do_sysconf (int name, const char * str)
{
  int rc = 0;
  long val;
  errno = 0;
  val = sysconf (name);
  if (val == -1) {
    if (errno != EINVAL) {
      printf("error: sysconf(%s): unexpected errno(%d)\n", str, errno);
      exit (1);
    }
    printf ("info: sysconf(%s): unsupported\n", str);
    rc = 1;
  } else
    printf ("sysconf(%s) = 0x%lx (%ld)\n", str, val, val);
  return rc;
}

static int
do_test (void)
{
  int rc = 0;

  rc += call_str (do_sysconf, _SC_LEVEL1_ICACHE_SIZE);
  rc += call_str (do_sysconf, _SC_LEVEL1_ICACHE_ASSOC);
  rc += call_str (do_sysconf, _SC_LEVEL1_ICACHE_LINESIZE);
  rc += call_str (do_sysconf, _SC_LEVEL1_DCACHE_SIZE);
  rc += call_str (do_sysconf, _SC_LEVEL1_DCACHE_ASSOC);
  rc += call_str (do_sysconf, _SC_LEVEL1_DCACHE_LINESIZE);
  rc += call_str (do_sysconf, _SC_LEVEL2_CACHE_SIZE);
  rc += call_str (do_sysconf, _SC_LEVEL2_CACHE_ASSOC);
  rc += call_str (do_sysconf, _SC_LEVEL2_CACHE_LINESIZE);
  rc += call_str (do_sysconf, _SC_LEVEL3_CACHE_SIZE);
  rc += call_str (do_sysconf, _SC_LEVEL3_CACHE_ASSOC);
  rc += call_str (do_sysconf, _SC_LEVEL3_CACHE_LINESIZE);

  if (rc)
    return EXIT_UNSUPPORTED;
  return 0;
}

#include <support/test-driver.c>
