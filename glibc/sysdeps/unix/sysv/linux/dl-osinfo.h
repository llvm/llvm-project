/* Operating system specific code for generic dynamic loader functions.  Linux.
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

#include <dl-sysdep.h>
#include <endian.h>
#include <fcntl.h>
#include <stdint.h>
#include <not-cancel.h>

#ifndef MIN
# define MIN(a,b) (((a)<(b))?(a):(b))
#endif

#define DL_SYSDEP_OSCHECK(FATAL)					      \
  do {									      \
    /* Test whether the kernel is new enough.  This test is only performed    \
       if the library is not compiled to run on all kernels.  */	      \
									      \
    int version = _dl_discover_osversion ();				      \
    if (__glibc_likely (version >= 0))					      \
      {									      \
	if (__builtin_expect (GLRO(dl_osversion) == 0, 1)		      \
	    || GLRO(dl_osversion) > version)				      \
	  GLRO(dl_osversion) = version;					      \
									      \
	/* Now we can test with the required version.  */		      \
	if (__LINUX_KERNEL_VERSION > 0 && version < __LINUX_KERNEL_VERSION)   \
	  /* Not sufficent.  */						      \
	  FATAL ("FATAL: kernel too old\n");				      \
      }									      \
    else if (__LINUX_KERNEL_VERSION > 0)				      \
      FATAL ("FATAL: cannot determine kernel version\n");		      \
  } while (0)

static inline uintptr_t __attribute__ ((always_inline))
_dl_setup_stack_chk_guard (void *dl_random)
{
  union
  {
    uintptr_t num;
    unsigned char bytes[sizeof (uintptr_t)];
  } ret;

  /* We need in the moment only 8 bytes on 32-bit platforms and 16
     bytes on 64-bit platforms.  Therefore we can use the data
     directly and not use the kernel-provided data to seed a PRNG.  */
  memcpy (ret.bytes, dl_random, sizeof (ret));
#if BYTE_ORDER == LITTLE_ENDIAN
  ret.num &= ~(uintptr_t) 0xff;
#elif BYTE_ORDER == BIG_ENDIAN
  ret.num &= ~((uintptr_t) 0xff << (8 * (sizeof (ret) - 1)));
#else
# error "BYTE_ORDER unknown"
#endif
  return ret.num;
}

static inline uintptr_t __attribute__ ((always_inline))
_dl_setup_pointer_guard (void *dl_random, uintptr_t stack_chk_guard)
{
  uintptr_t ret;
  memcpy (&ret, (char *) dl_random + sizeof (ret), sizeof (ret));
  return ret;
}
