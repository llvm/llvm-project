/* Enumerate available IFUNC implementations of a function.  sparc version.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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
#include <sysdep.h>
#include <ifunc-impl-list.h>

/* Fill ARRAY of MAX elements with IFUNC implementations for function
   NAME and return the number of valid entries.  */

size_t
__libc_ifunc_impl_list (const char *name, struct libc_ifunc_impl *array,
			size_t max)
{
  size_t i = 0;
  int hwcap;

  hwcap = GLRO(dl_hwcap);

  IFUNC_IMPL (i, name, memcpy,
	      IFUNC_IMPL_ADD (array, i, memcpy, hwcap & HWCAP_SPARC_ADP,
			      __memcpy_niagara7)
	      IFUNC_IMPL_ADD (array, i, memcpy, hwcap & HWCAP_SPARC_CRYPTO,
			      __memcpy_niagara4)
	      IFUNC_IMPL_ADD (array, i, memcpy, hwcap & HWCAP_SPARC_N2,
			      __memcpy_niagara2)
	      IFUNC_IMPL_ADD (array, i, memcpy, hwcap & HWCAP_SPARC_BLKINIT,
			      __memcpy_niagara1)
	      IFUNC_IMPL_ADD (array, i, memcpy, hwcap & HWCAP_SPARC_ULTRA3,
			      __memcpy_ultra3)
	      IFUNC_IMPL_ADD (array, i, memcpy, 1, __memcpy_ultra1));

  IFUNC_IMPL (i, name, mempcpy,
	      IFUNC_IMPL_ADD (array, i, mempcpy, hwcap & HWCAP_SPARC_ADP,
			      __mempcpy_niagara7)
	      IFUNC_IMPL_ADD (array, i, mempcpy, hwcap & HWCAP_SPARC_CRYPTO,
			      __mempcpy_niagara4)
	      IFUNC_IMPL_ADD (array, i, mempcpy, hwcap & HWCAP_SPARC_N2,
			      __mempcpy_niagara2)
	      IFUNC_IMPL_ADD (array, i, mempcpy, hwcap & HWCAP_SPARC_BLKINIT,
			      __mempcpy_niagara1)
	      IFUNC_IMPL_ADD (array, i, mempcpy, hwcap & HWCAP_SPARC_ULTRA3,
			      __mempcpy_ultra3)
	      IFUNC_IMPL_ADD (array, i, mempcpy, 1, __mempcpy_ultra1));

  IFUNC_IMPL (i, name, bzero,
	      IFUNC_IMPL_ADD (array, i, bzero, hwcap & HWCAP_SPARC_ADP,
			      __bzero_niagara7)
	      IFUNC_IMPL_ADD (array, i, bzero, hwcap & HWCAP_SPARC_CRYPTO,
			      __bzero_niagara4)
	      IFUNC_IMPL_ADD (array, i, bzero, hwcap & HWCAP_SPARC_BLKINIT,
			      __bzero_niagara1)
	      IFUNC_IMPL_ADD (array, i, bzero, 1, __bzero_ultra1));

  IFUNC_IMPL (i, name, memset,
	      IFUNC_IMPL_ADD (array, i, memset, hwcap & HWCAP_SPARC_ADP,
			      __memset_niagara7)
	      IFUNC_IMPL_ADD (array, i, memset, hwcap & HWCAP_SPARC_CRYPTO,
			      __memset_niagara4)
	      IFUNC_IMPL_ADD (array, i, memset, hwcap & HWCAP_SPARC_BLKINIT,
			      __memset_niagara1)
	      IFUNC_IMPL_ADD (array, i, memset, 1, __memset_ultra1));

  IFUNC_IMPL (i, name, memmove,
	      IFUNC_IMPL_ADD (array, i, memmove, hwcap & HWCAP_SPARC_ADP,
			      __memmove_niagara7)
	      IFUNC_IMPL_ADD (array, i, memmove, 1, __memmove_ultra1));

  return i;
}
