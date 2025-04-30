/* Enumerate available IFUNC implementations of a function.  PowerPC32 version.
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

#include <assert.h>
#include <string.h>
#include <wchar.h>
#include <ldsodefs.h>
#include <ifunc-impl-list.h>

/* Maximum number of IFUNC implementations.  */
#define MAX_IFUNC	6

size_t
__libc_ifunc_impl_list (const char *name, struct libc_ifunc_impl *array,
			size_t max)
{
  assert (max >= MAX_IFUNC);

  size_t i = 0;

  unsigned long int hwcap = GLRO(dl_hwcap);
  /* hwcap contains only the latest supported ISA, the code checks which is
     and fills the previous supported ones.  */
  if (hwcap & PPC_FEATURE_ARCH_2_06)
    hwcap |= PPC_FEATURE_ARCH_2_05 | PPC_FEATURE_POWER5_PLUS |
             PPC_FEATURE_POWER5 | PPC_FEATURE_POWER4;
  else if (hwcap & PPC_FEATURE_ARCH_2_05)
    hwcap |= PPC_FEATURE_POWER5_PLUS | PPC_FEATURE_POWER5 | PPC_FEATURE_POWER4;
  else if (hwcap & PPC_FEATURE_POWER5_PLUS)
    hwcap |= PPC_FEATURE_POWER5 | PPC_FEATURE_POWER4;
  else if (hwcap & PPC_FEATURE_POWER5)
    hwcap |= PPC_FEATURE_POWER4;

#ifdef SHARED
  /* Support sysdeps/powerpc/powerpc32/power4/multiarch/memcpy.c.  */
  IFUNC_IMPL (i, name, memcpy,
	      IFUNC_IMPL_ADD (array, i, memcpy, hwcap & PPC_FEATURE_HAS_VSX,
			      __memcpy_power7)
	      IFUNC_IMPL_ADD (array, i, memcpy, hwcap & PPC_FEATURE_ARCH_2_06,
			      __memcpy_a2)
	      IFUNC_IMPL_ADD (array, i, memcpy, hwcap & PPC_FEATURE_ARCH_2_05,
			      __memcpy_power6)
	      IFUNC_IMPL_ADD (array, i, memcpy, hwcap & PPC_FEATURE_CELL_BE,
			      __memcpy_cell)
	      IFUNC_IMPL_ADD (array, i, memcpy, 1, __memcpy_ppc))

  /* Support sysdeps/powerpc/powerpc32/power4/multiarch/memmove.c.  */
  IFUNC_IMPL (i, name, memmove,
	      IFUNC_IMPL_ADD (array, i, memmove, hwcap & PPC_FEATURE_HAS_VSX,
			      __memmove_power7)
	      IFUNC_IMPL_ADD (array, i, memmove, 1, __memmove_ppc))

  /* Support sysdeps/powerpc/powerpc32/power4/multiarch/memset.c.  */
  IFUNC_IMPL (i, name, memset,
	      IFUNC_IMPL_ADD (array, i, memset, hwcap & PPC_FEATURE_HAS_VSX,
			      __memset_power7)
	      IFUNC_IMPL_ADD (array, i, memset, hwcap & PPC_FEATURE_ARCH_2_05,
			      __memset_power6)
	      IFUNC_IMPL_ADD (array, i, memset, 1, __memset_ppc))

  /* Support sysdeps/powerpc/powerpc32/power4/multiarch/bzero.c.  */
  IFUNC_IMPL (i, name, bzero,
	      IFUNC_IMPL_ADD (array, i, bzero, hwcap & PPC_FEATURE_HAS_VSX,
			      __bzero_power7)
	      IFUNC_IMPL_ADD (array, i, bzero, hwcap & PPC_FEATURE_ARCH_2_05,
			      __bzero_power6)
	      IFUNC_IMPL_ADD (array, i, bzero, 1, __bzero_ppc))

  /* Support sysdeps/powerpc/powerpc32/power4/multiarch/strlen.c.  */
  IFUNC_IMPL (i, name, strlen,
	      IFUNC_IMPL_ADD (array, i, strlen, hwcap & PPC_FEATURE_HAS_VSX,
			      __strlen_power7)
	      IFUNC_IMPL_ADD (array, i, strlen, 1,
			      __strlen_ppc))

  /* Support sysdeps/powerpc/powerpc32/power4/multiarch/strnlen.c.  */
  IFUNC_IMPL (i, name, strnlen,
	      IFUNC_IMPL_ADD (array, i, strnlen, hwcap & PPC_FEATURE_HAS_VSX,
			      __strnlen_power7)
	      IFUNC_IMPL_ADD (array, i, strnlen, 1,
			      __strnlen_ppc))

  /* Support sysdeps/powerpc/powerpc32/multiarch/strncmp.c.  */
  IFUNC_IMPL (i, name, strncmp,
	      IFUNC_IMPL_ADD (array, i, strncmp, hwcap & PPC_FEATURE_HAS_VSX,
			      __strncmp_power7)
	      IFUNC_IMPL_ADD (array, i, strncmp, 1,
			      __strncmp_ppc))
#endif

  /* Support sysdeps/powerpc/powerpc32/power4/multiarch/memcmp.c.  */
  IFUNC_IMPL (i, name, memcmp,
	      IFUNC_IMPL_ADD (array, i, memcmp, hwcap & PPC_FEATURE_HAS_VSX,
			      __memcmp_power7)
	      IFUNC_IMPL_ADD (array, i, memcmp, 1, __memcmp_ppc))

  /* Support sysdeps/powerpc/powerpc32/power4/multiarch/mempcpy.c.  */
  IFUNC_IMPL (i, name, mempcpy,
	      IFUNC_IMPL_ADD (array, i, mempcpy,
			      hwcap & PPC_FEATURE_HAS_VSX,
			      __mempcpy_power7)
	      IFUNC_IMPL_ADD (array, i, mempcpy, 1,
			      __mempcpy_ppc))

  /* Support sysdeps/powerpc/powerpc32/power4/multiarch/memchr.c.  */
  IFUNC_IMPL (i, name, memchr,
	      IFUNC_IMPL_ADD (array, i, memchr,
			      hwcap & PPC_FEATURE_HAS_VSX,
			      __memchr_power7)
	      IFUNC_IMPL_ADD (array, i, memchr, 1,
			      __memchr_ppc))

  /* Support sysdeps/powerpc/powerpc32/power4/multiarch/memrchr.c.  */
  IFUNC_IMPL (i, name, memrchr,
	      IFUNC_IMPL_ADD (array, i, memrchr,
			      hwcap & PPC_FEATURE_HAS_VSX,
			      __memrchr_power7)
	      IFUNC_IMPL_ADD (array, i, memrchr, 1,
			      __memrchr_ppc))

  /* Support sysdeps/powerpc/powerpc32/power4/multiarch/rawmemchr.c.  */
  IFUNC_IMPL (i, name, rawmemchr,
	      IFUNC_IMPL_ADD (array, i, rawmemchr,
			      hwcap & PPC_FEATURE_HAS_VSX,
			      __rawmemchr_power7)
	      IFUNC_IMPL_ADD (array, i, rawmemchr, 1,
			      __rawmemchr_ppc))

  /* Support sysdeps/powerpc/powerpc32/power4/multiarch/strcasecmp.c.  */
  IFUNC_IMPL (i, name, strcasecmp,
	      IFUNC_IMPL_ADD (array, i, strcasecmp,
			      hwcap & PPC_FEATURE_HAS_VSX,
			      __strcasecmp_power7)
	      IFUNC_IMPL_ADD (array, i, strcasecmp, 1, __strcasecmp_ppc))

  /* Support sysdeps/powerpc/powerpc32/power4/multiarch/strcasecmp_l.c.  */
  IFUNC_IMPL (i, name, strcasecmp_l,
	      IFUNC_IMPL_ADD (array, i, strcasecmp_l,
			      hwcap & PPC_FEATURE_HAS_VSX,
			      __strcasecmp_l_power7)
	      IFUNC_IMPL_ADD (array, i, strcasecmp_l, 1,
			      __strcasecmp_l_ppc))

  /* Support sysdeps/powerpc/powerpc32/power4/multiarch/strncase.c.  */
  IFUNC_IMPL (i, name, strncasecmp,
	      IFUNC_IMPL_ADD (array, i, strncasecmp,
			      hwcap & PPC_FEATURE_HAS_VSX,
			      __strncasecmp_power7)
	      IFUNC_IMPL_ADD (array, i, strncasecmp, 1, __strncasecmp_ppc))

  /* Support sysdeps/powerpc/powerpc32/power4/multiarch/strncase_l.c.  */
  IFUNC_IMPL (i, name, strncasecmp_l,
	      IFUNC_IMPL_ADD (array, i, strncasecmp_l,
			      hwcap & PPC_FEATURE_HAS_VSX,
			      __strncasecmp_l_power7)
	      IFUNC_IMPL_ADD (array, i, strncasecmp_l, 1,
			      __strncasecmp_l_ppc))

  /* Support sysdeps/powerpc/powerpc32/power4/multiarch/strchrnul.c.  */
  IFUNC_IMPL (i, name, strchrnul,
	      IFUNC_IMPL_ADD (array, i, strchrnul,
			      hwcap & PPC_FEATURE_HAS_VSX,
			      __strchrnul_power7)
	      IFUNC_IMPL_ADD (array, i, strchrnul, 1,
			      __strchrnul_ppc))

  /* Support sysdeps/powerpc/powerpc32/power4/multiarch/strchr.c.  */
  IFUNC_IMPL (i, name, strchr,
	      IFUNC_IMPL_ADD (array, i, strchr,
			      hwcap & PPC_FEATURE_HAS_VSX,
			      __strchr_power7)
	      IFUNC_IMPL_ADD (array, i, strchr, 1,
			      __strchr_ppc))

  return i;
}
