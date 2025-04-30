/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
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

#define CSHAPE(totalsize, linesize, assoc) \
  ((totalsize & ~0xff) | (linesize << 4) | assoc)

extern long __libc_alpha_cache_shape[4];

/* Get the value of the system variable NAME.  */
long int
__sysconf (int name)
{
  long shape, index;

  /* We only handle the cache information here (for now).  */
  if (name < _SC_LEVEL1_ICACHE_SIZE || name > _SC_LEVEL4_CACHE_LINESIZE)
    return linux_sysconf (name);

  /* No Alpha has L4 caches.  */
  if (name >= _SC_LEVEL4_CACHE_SIZE)
    return -1;

  index = (name - _SC_LEVEL1_ICACHE_SIZE) / 3;
  shape = __libc_alpha_cache_shape[index];
  if (shape == -2)
    {
      long shape_l1i, shape_l1d, shape_l2, shape_l3 = -1;

      /* ??? In the cases below for which we do not know L1 cache sizes,
	 we could do timings to measure sizes.  But for the Bcache, it's
	 generally big enough that (without additional help) TLB effects
	 get in the way.  We'd either need to be able to allocate large
	 pages or have the kernel do the timings from KSEG.  Fortunately,
	 kernels beginning with 2.6.5 will pass us this info in auxvec.  */

      switch (__builtin_alpha_implver ())
	{
	case 0: /* EV4 */
	  /* EV4/LCA45 had 8k L1 caches; EV45 had 16k L1 caches.  */
	  /* EV4/EV45 had 128k to 16M 32-byte direct Bcache.  LCA45
	     had 64k to 8M 8-byte direct Bcache.  Can't tell.  */
	  shape_l1i = shape_l1d = shape_l2 = CSHAPE (0, 5, 1);
	  break;

	case 1: /* EV5 */
	  if (__builtin_alpha_amask (1 << 8))
	    {
	      /* MAX insns not present; either EV5 or EV56.  */
	      shape_l1i = shape_l1d = CSHAPE(8*1024, 5, 1);
	      /* ??? L2 and L3 *can* be configured as 32-byte line.  */
	      shape_l2 = CSHAPE (96*1024, 6, 3);
	      /* EV5/EV56 has 1M to 16M Bcache.  */
	      shape_l3 = CSHAPE (0, 6, 1);
	    }
	  else
	    {
	      /* MAX insns present; either PCA56 or PCA57.  */
	      /* PCA56 had 16k 64-byte cache; PCA57 had 32k Icache.  */
	      /* PCA56 had 8k 64-byte cache; PCA57 had 16k Dcache.  */
	      /* PCA5[67] had 512k to 4M Bcache.  */
	      shape_l1i = shape_l1d = shape_l2 = CSHAPE (0, 6, 1);
	    }
	  break;

	case 2: /* EV6 */
	  shape_l1i = shape_l1d = CSHAPE(64*1024, 6, 2);
	  /* EV6/EV67/EV68* had 1M to 16M Bcache.  */
	  shape_l2 = CSHAPE (0, 6, 1);
	  break;

	case 3: /* EV7 */
	  shape_l1i = shape_l1d = CSHAPE(64*1024, 6, 2);
	  shape_l2 = CSHAPE(7*1024*1024/4, 6, 7);
	  break;

	default:
	  shape_l1i = shape_l1d = shape_l2 = 0;
	  break;
	}

      __libc_alpha_cache_shape[0] = shape_l1i;
      __libc_alpha_cache_shape[1] = shape_l1d;
      __libc_alpha_cache_shape[2] = shape_l2;
      __libc_alpha_cache_shape[3] = shape_l3;
      shape = __libc_alpha_cache_shape[index];
    }

  if (shape <= 0)
    return shape;

  switch ((name - _SC_LEVEL1_ICACHE_SIZE) % 3)
    {
    case 0: /* total size */
      return shape & -0x100;
    case 1: /* associativity */
      return shape & 0xf;
    default: /* line size */
      return 1L << ((shape >> 4) & 0xf);
    }
}

/* Now the generic Linux version.  */
#undef __sysconf
#define __sysconf static linux_sysconf
#include <sysdeps/unix/sysv/linux/sysconf.c>
