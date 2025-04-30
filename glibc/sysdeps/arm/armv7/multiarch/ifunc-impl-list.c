/* Enumerate available IFUNC implementations of a function.  ARM version.
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

#include <stdbool.h>
#include <string.h>
#include <ldsodefs.h>
#include <sysdep.h>
#include <ifunc-impl-list.h>

/* Fill ARRAY of MAX elements with IFUNC implementations for function
   NAME and return the number of valid entries.  */

size_t
__libc_ifunc_impl_list (const char *name, struct libc_ifunc_impl *array,
			size_t max)
{
  size_t i = 0;

  bool use_neon = true;
#ifdef __ARM_NEON__
# define __memcpy_neon	memcpy
# define __memchr_neon	memchr
#else
  use_neon = (GLRO(dl_hwcap) & HWCAP_ARM_NEON) != 0;
#endif

#ifndef __ARM_NEON__
  bool use_vfp = true;
# ifdef __SOFTFP__
  use_vfp = (GLRO(dl_hwcap) & HWCAP_ARM_VFP) != 0;
# endif
#endif

  IFUNC_IMPL (i, name, memcpy,
	      IFUNC_IMPL_ADD (array, i, memcpy, use_neon, __memcpy_neon)
#ifndef __ARM_NEON__
	      IFUNC_IMPL_ADD (array, i, memcpy, use_vfp, __memcpy_vfp)
#endif
	      IFUNC_IMPL_ADD (array, i, memcpy, 1, __memcpy_arm));

  IFUNC_IMPL (i, name, memchr,
	      IFUNC_IMPL_ADD (array, i, memchr, use_neon, __memchr_neon)
	      IFUNC_IMPL_ADD (array, i, memchr, 1, __memchr_noneon));

  return i;
}
