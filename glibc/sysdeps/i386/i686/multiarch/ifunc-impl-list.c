/* Enumerate available IFUNC implementations of a function.  i686 version.
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
#include <ifunc-impl-list.h>
#include "init-arch.h"

/* Maximum number of IFUNC implementations.  */
#define MAX_IFUNC	4

/* Fill ARRAY of MAX elements with IFUNC implementations for function
   NAME and return the number of valid entries.  */

size_t
__libc_ifunc_impl_list (const char *name, struct libc_ifunc_impl *array,
			size_t max)
{
  assert (max >= MAX_IFUNC);

  size_t i = 0;

  /* Support sysdeps/i386/i686/multiarch/bcopy.S.  */
  IFUNC_IMPL (i, name, bcopy,
	      IFUNC_IMPL_ADD (array, i, bcopy, CPU_FEATURE_USABLE (SSSE3),
			      __bcopy_ssse3_rep)
	      IFUNC_IMPL_ADD (array, i, bcopy, CPU_FEATURE_USABLE (SSSE3),
			      __bcopy_ssse3)
	      IFUNC_IMPL_ADD (array, i, bcopy, CPU_FEATURE_USABLE (SSE2),
			      __bcopy_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, bcopy, 1, __bcopy_ia32))

  /* Support sysdeps/i386/i686/multiarch/bzero.S.  */
  IFUNC_IMPL (i, name, bzero,
	      IFUNC_IMPL_ADD (array, i, bzero, CPU_FEATURE_USABLE (SSE2),
			      __bzero_sse2_rep)
	      IFUNC_IMPL_ADD (array, i, bzero, CPU_FEATURE_USABLE (SSE2),
			      __bzero_sse2)
	      IFUNC_IMPL_ADD (array, i, bzero, 1, __bzero_ia32))

  /* Support sysdeps/i386/i686/multiarch/memchr.S.  */
  IFUNC_IMPL (i, name, memchr,
	      IFUNC_IMPL_ADD (array, i, memchr, CPU_FEATURE_USABLE (SSE2),
			      __memchr_sse2_bsf)
	      IFUNC_IMPL_ADD (array, i, memchr, CPU_FEATURE_USABLE (SSE2),
			      __memchr_sse2)
	      IFUNC_IMPL_ADD (array, i, memchr, 1, __memchr_ia32))

  /* Support sysdeps/i386/i686/multiarch/memcmp.S.  */
  IFUNC_IMPL (i, name, memcmp,
	      IFUNC_IMPL_ADD (array, i, memcmp, CPU_FEATURE_USABLE (SSE4_2),
			      __memcmp_sse4_2)
	      IFUNC_IMPL_ADD (array, i, memcmp, CPU_FEATURE_USABLE (SSSE3),
			      __memcmp_ssse3)
	      IFUNC_IMPL_ADD (array, i, memcmp, 1, __memcmp_ia32))

#ifdef SHARED
  /* Support sysdeps/i386/i686/multiarch/memmove_chk.S.  */
  IFUNC_IMPL (i, name, __memmove_chk,
	      IFUNC_IMPL_ADD (array, i, __memmove_chk,
			      CPU_FEATURE_USABLE (SSSE3),
			      __memmove_chk_ssse3_rep)
	      IFUNC_IMPL_ADD (array, i, __memmove_chk,
			      CPU_FEATURE_USABLE (SSSE3),
			      __memmove_chk_ssse3)
	      IFUNC_IMPL_ADD (array, i, __memmove_chk,
			      CPU_FEATURE_USABLE (SSE2),
			      __memmove_chk_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, __memmove_chk, 1,
			      __memmove_chk_ia32))
#endif

  /* Support sysdeps/i386/i686/multiarch/memmove.S.  */
  IFUNC_IMPL (i, name, memmove,
	      IFUNC_IMPL_ADD (array, i, memmove, CPU_FEATURE_USABLE (SSSE3),
			      __memmove_ssse3_rep)
	      IFUNC_IMPL_ADD (array, i, memmove, CPU_FEATURE_USABLE (SSSE3),
			      __memmove_ssse3)
	      IFUNC_IMPL_ADD (array, i, memmove, CPU_FEATURE_USABLE (SSE2),
			      __memmove_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, memmove, 1, __memmove_ia32))

  /* Support sysdeps/i386/i686/multiarch/memrchr.S.  */
  IFUNC_IMPL (i, name, memrchr,
	      IFUNC_IMPL_ADD (array, i, memrchr, CPU_FEATURE_USABLE (SSE2),
			      __memrchr_sse2_bsf)
	      IFUNC_IMPL_ADD (array, i, memrchr, CPU_FEATURE_USABLE (SSE2),
			      __memrchr_sse2)
	      IFUNC_IMPL_ADD (array, i, memrchr, 1, __memrchr_ia32))

#ifdef SHARED
  /* Support sysdeps/i386/i686/multiarch/memset_chk.S.  */
  IFUNC_IMPL (i, name, __memset_chk,
	      IFUNC_IMPL_ADD (array, i, __memset_chk,
			      CPU_FEATURE_USABLE (SSE2),
			      __memset_chk_sse2_rep)
	      IFUNC_IMPL_ADD (array, i, __memset_chk,
			      CPU_FEATURE_USABLE (SSE2),
			      __memset_chk_sse2)
	      IFUNC_IMPL_ADD (array, i, __memset_chk, 1,
			      __memset_chk_ia32))
#endif

  /* Support sysdeps/i386/i686/multiarch/memset.S.  */
  IFUNC_IMPL (i, name, memset,
	      IFUNC_IMPL_ADD (array, i, memset, CPU_FEATURE_USABLE (SSE2),
			      __memset_sse2_rep)
	      IFUNC_IMPL_ADD (array, i, memset, CPU_FEATURE_USABLE (SSE2),
			      __memset_sse2)
	      IFUNC_IMPL_ADD (array, i, memset, 1, __memset_ia32))

  /* Support sysdeps/i386/i686/multiarch/rawmemchr.S.  */
  IFUNC_IMPL (i, name, rawmemchr,
	      IFUNC_IMPL_ADD (array, i, rawmemchr, CPU_FEATURE_USABLE (SSE2),
			      __rawmemchr_sse2_bsf)
	      IFUNC_IMPL_ADD (array, i, rawmemchr, CPU_FEATURE_USABLE (SSE2),
			      __rawmemchr_sse2)
	      IFUNC_IMPL_ADD (array, i, rawmemchr, 1, __rawmemchr_ia32))

  /* Support sysdeps/i386/i686/multiarch/stpncpy.S.  */
  IFUNC_IMPL (i, name, stpncpy,
	      IFUNC_IMPL_ADD (array, i, stpncpy, CPU_FEATURE_USABLE (SSSE3),
			      __stpncpy_ssse3)
	      IFUNC_IMPL_ADD (array, i, stpncpy, CPU_FEATURE_USABLE (SSE2),
			      __stpncpy_sse2)
	      IFUNC_IMPL_ADD (array, i, stpncpy, 1, __stpncpy_ia32))

  /* Support sysdeps/i386/i686/multiarch/stpcpy.S.  */
  IFUNC_IMPL (i, name, stpcpy,
	      IFUNC_IMPL_ADD (array, i, stpcpy, CPU_FEATURE_USABLE (SSSE3),
			      __stpcpy_ssse3)
	      IFUNC_IMPL_ADD (array, i, stpcpy, CPU_FEATURE_USABLE (SSE2),
			      __stpcpy_sse2)
	      IFUNC_IMPL_ADD (array, i, stpcpy, 1, __stpcpy_ia32))

  /* Support sysdeps/i386/i686/multiarch/strcasecmp.S.  */
  IFUNC_IMPL (i, name, strcasecmp,
	      IFUNC_IMPL_ADD (array, i, strcasecmp,
			      CPU_FEATURE_USABLE (SSE4_2),
			      __strcasecmp_sse4_2)
	      IFUNC_IMPL_ADD (array, i, strcasecmp,
			      CPU_FEATURE_USABLE (SSSE3),
			      __strcasecmp_ssse3)
	      IFUNC_IMPL_ADD (array, i, strcasecmp, 1, __strcasecmp_ia32))

  /* Support sysdeps/i386/i686/multiarch/strcasecmp_l.S.  */
  IFUNC_IMPL (i, name, strcasecmp_l,
	      IFUNC_IMPL_ADD (array, i, strcasecmp_l,
			      CPU_FEATURE_USABLE (SSE4_2),
			      __strcasecmp_l_sse4_2)
	      IFUNC_IMPL_ADD (array, i, strcasecmp_l,
			      CPU_FEATURE_USABLE (SSSE3),
			      __strcasecmp_l_ssse3)
	      IFUNC_IMPL_ADD (array, i, strcasecmp_l, 1,
			      __strcasecmp_l_ia32))

  /* Support sysdeps/i386/i686/multiarch/strcat.S.  */
  IFUNC_IMPL (i, name, strcat,
	      IFUNC_IMPL_ADD (array, i, strcat, CPU_FEATURE_USABLE (SSSE3),
			      __strcat_ssse3)
	      IFUNC_IMPL_ADD (array, i, strcat, CPU_FEATURE_USABLE (SSE2),
			      __strcat_sse2)
	      IFUNC_IMPL_ADD (array, i, strcat, 1, __strcat_ia32))

  /* Support sysdeps/i386/i686/multiarch/strchr.S.  */
  IFUNC_IMPL (i, name, strchr,
	      IFUNC_IMPL_ADD (array, i, strchr, CPU_FEATURE_USABLE (SSE2),
			      __strchr_sse2_bsf)
	      IFUNC_IMPL_ADD (array, i, strchr, CPU_FEATURE_USABLE (SSE2),
			      __strchr_sse2)
	      IFUNC_IMPL_ADD (array, i, strchr, 1, __strchr_ia32))

  /* Support sysdeps/i386/i686/multiarch/strcmp.S.  */
  IFUNC_IMPL (i, name, strcmp,
	      IFUNC_IMPL_ADD (array, i, strcmp, CPU_FEATURE_USABLE (SSE4_2),
			      __strcmp_sse4_2)
	      IFUNC_IMPL_ADD (array, i, strcmp, CPU_FEATURE_USABLE (SSSE3),
			      __strcmp_ssse3)
	      IFUNC_IMPL_ADD (array, i, strcmp, 1, __strcmp_ia32))

  /* Support sysdeps/i386/i686/multiarch/strcpy.S.  */
  IFUNC_IMPL (i, name, strcpy,
	      IFUNC_IMPL_ADD (array, i, strcpy, CPU_FEATURE_USABLE (SSSE3),
			      __strcpy_ssse3)
	      IFUNC_IMPL_ADD (array, i, strcpy, CPU_FEATURE_USABLE (SSE2),
			      __strcpy_sse2)
	      IFUNC_IMPL_ADD (array, i, strcpy, 1, __strcpy_ia32))

  /* Support sysdeps/i386/i686/multiarch/strcspn.S.  */
  IFUNC_IMPL (i, name, strcspn,
	      IFUNC_IMPL_ADD (array, i, strcspn, CPU_FEATURE_USABLE (SSE4_2),
			      __strcspn_sse42)
	      IFUNC_IMPL_ADD (array, i, strcspn, 1, __strcspn_ia32))

  /* Support sysdeps/i386/i686/multiarch/strncase.S.  */
  IFUNC_IMPL (i, name, strncasecmp,
	      IFUNC_IMPL_ADD (array, i, strncasecmp,
			      CPU_FEATURE_USABLE (SSE4_2),
			      __strncasecmp_sse4_2)
	      IFUNC_IMPL_ADD (array, i, strncasecmp,
			      CPU_FEATURE_USABLE (SSSE3),
			      __strncasecmp_ssse3)
	      IFUNC_IMPL_ADD (array, i, strncasecmp, 1,
			      __strncasecmp_ia32))

  /* Support sysdeps/i386/i686/multiarch/strncase_l.S.  */
  IFUNC_IMPL (i, name, strncasecmp_l,
	      IFUNC_IMPL_ADD (array, i, strncasecmp_l,
			      CPU_FEATURE_USABLE (SSE4_2),
			      __strncasecmp_l_sse4_2)
	      IFUNC_IMPL_ADD (array, i, strncasecmp_l,
			      CPU_FEATURE_USABLE (SSSE3),
			      __strncasecmp_l_ssse3)
	      IFUNC_IMPL_ADD (array, i, strncasecmp_l, 1,
			      __strncasecmp_l_ia32))

  /* Support sysdeps/i386/i686/multiarch/strncat.S.  */
  IFUNC_IMPL (i, name, strncat,
	      IFUNC_IMPL_ADD (array, i, strncat, CPU_FEATURE_USABLE (SSSE3),
			      __strncat_ssse3)
	      IFUNC_IMPL_ADD (array, i, strncat, CPU_FEATURE_USABLE (SSE2),
			      __strncat_sse2)
	      IFUNC_IMPL_ADD (array, i, strncat, 1, __strncat_ia32))

  /* Support sysdeps/i386/i686/multiarch/strncpy.S.  */
  IFUNC_IMPL (i, name, strncpy,
	      IFUNC_IMPL_ADD (array, i, strncpy, CPU_FEATURE_USABLE (SSSE3),
			      __strncpy_ssse3)
	      IFUNC_IMPL_ADD (array, i, strncpy, CPU_FEATURE_USABLE (SSE2),
			      __strncpy_sse2)
	      IFUNC_IMPL_ADD (array, i, strncpy, 1, __strncpy_ia32))

  /* Support sysdeps/i386/i686/multiarch/strnlen.S.  */
  IFUNC_IMPL (i, name, strnlen,
	      IFUNC_IMPL_ADD (array, i, strnlen, CPU_FEATURE_USABLE (SSE2),
			      __strnlen_sse2)
	      IFUNC_IMPL_ADD (array, i, strnlen, 1, __strnlen_ia32))

  /* Support sysdeps/i386/i686/multiarch/strpbrk.S.  */
  IFUNC_IMPL (i, name, strpbrk,
	      IFUNC_IMPL_ADD (array, i, strpbrk, CPU_FEATURE_USABLE (SSE4_2),
			      __strpbrk_sse42)
	      IFUNC_IMPL_ADD (array, i, strpbrk, 1, __strpbrk_ia32))

  /* Support sysdeps/i386/i686/multiarch/strrchr.S.  */
  IFUNC_IMPL (i, name, strrchr,
	      IFUNC_IMPL_ADD (array, i, strrchr, CPU_FEATURE_USABLE (SSE2),
			      __strrchr_sse2_bsf)
	      IFUNC_IMPL_ADD (array, i, strrchr, CPU_FEATURE_USABLE (SSE2),
			      __strrchr_sse2)
	      IFUNC_IMPL_ADD (array, i, strrchr, 1, __strrchr_ia32))

  /* Support sysdeps/i386/i686/multiarch/strspn.S.  */
  IFUNC_IMPL (i, name, strspn,
	      IFUNC_IMPL_ADD (array, i, strspn, CPU_FEATURE_USABLE (SSE4_2),
			      __strspn_sse42)
	      IFUNC_IMPL_ADD (array, i, strspn, 1, __strspn_ia32))

  /* Support sysdeps/i386/i686/multiarch/wcschr.S.  */
  IFUNC_IMPL (i, name, wcschr,
	      IFUNC_IMPL_ADD (array, i, wcschr, CPU_FEATURE_USABLE (SSE2),
			      __wcschr_sse2)
	      IFUNC_IMPL_ADD (array, i, wcschr, 1, __wcschr_ia32))

  /* Support sysdeps/i386/i686/multiarch/wcscmp.S.  */
  IFUNC_IMPL (i, name, wcscmp,
	      IFUNC_IMPL_ADD (array, i, wcscmp, CPU_FEATURE_USABLE (SSE2),
			      __wcscmp_sse2)
	      IFUNC_IMPL_ADD (array, i, wcscmp, 1, __wcscmp_ia32))

  /* Support sysdeps/i386/i686/multiarch/wcscpy.S.  */
  IFUNC_IMPL (i, name, wcscpy,
	      IFUNC_IMPL_ADD (array, i, wcscpy, CPU_FEATURE_USABLE (SSSE3),
			      __wcscpy_ssse3)
	      IFUNC_IMPL_ADD (array, i, wcscpy, 1, __wcscpy_ia32))

  /* Support sysdeps/i386/i686/multiarch/wcslen.S.  */
  IFUNC_IMPL (i, name, wcslen,
	      IFUNC_IMPL_ADD (array, i, wcslen, CPU_FEATURE_USABLE (SSE2),
			      __wcslen_sse2)
	      IFUNC_IMPL_ADD (array, i, wcslen, 1, __wcslen_ia32))

  /* Support sysdeps/i386/i686/multiarch/wcsrchr.S.  */
  IFUNC_IMPL (i, name, wcsrchr,
	      IFUNC_IMPL_ADD (array, i, wcsrchr, CPU_FEATURE_USABLE (SSE2),
			      __wcsrchr_sse2)
	      IFUNC_IMPL_ADD (array, i, wcsrchr, 1, __wcsrchr_ia32))

  /* Support sysdeps/i386/i686/multiarch/wmemcmp.S.  */
  IFUNC_IMPL (i, name, wmemcmp,
	      IFUNC_IMPL_ADD (array, i, wmemcmp, CPU_FEATURE_USABLE (SSE4_2),
			      __wmemcmp_sse4_2)
	      IFUNC_IMPL_ADD (array, i, wmemcmp, CPU_FEATURE_USABLE (SSSE3),
			      __wmemcmp_ssse3)
	      IFUNC_IMPL_ADD (array, i, wmemcmp, 1, __wmemcmp_ia32))

#ifdef SHARED
  /* Support sysdeps/i386/i686/multiarch/memcpy_chk.S.  */
  IFUNC_IMPL (i, name, __memcpy_chk,
	      IFUNC_IMPL_ADD (array, i, __memcpy_chk,
			      CPU_FEATURE_USABLE (SSSE3),
			      __memcpy_chk_ssse3_rep)
	      IFUNC_IMPL_ADD (array, i, __memcpy_chk,
			      CPU_FEATURE_USABLE (SSSE3),
			      __memcpy_chk_ssse3)
	      IFUNC_IMPL_ADD (array, i, __memcpy_chk,
			      CPU_FEATURE_USABLE (SSE2),
			      __memcpy_chk_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, __memcpy_chk, 1,
			      __memcpy_chk_ia32))

  /* Support sysdeps/i386/i686/multiarch/memcpy.S.  */
  IFUNC_IMPL (i, name, memcpy,
	      IFUNC_IMPL_ADD (array, i, memcpy, CPU_FEATURE_USABLE (SSSE3),
			      __memcpy_ssse3_rep)
	      IFUNC_IMPL_ADD (array, i, memcpy, CPU_FEATURE_USABLE (SSSE3),
			      __memcpy_ssse3)
	      IFUNC_IMPL_ADD (array, i, memcpy, CPU_FEATURE_USABLE (SSE2),
			      __memcpy_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, memcpy, 1, __memcpy_ia32))

  /* Support sysdeps/i386/i686/multiarch/mempcpy_chk.S.  */
  IFUNC_IMPL (i, name, __mempcpy_chk,
	      IFUNC_IMPL_ADD (array, i, __mempcpy_chk,
			      CPU_FEATURE_USABLE (SSSE3),
			      __mempcpy_chk_ssse3_rep)
	      IFUNC_IMPL_ADD (array, i, __mempcpy_chk,
			      CPU_FEATURE_USABLE (SSSE3),
			      __mempcpy_chk_ssse3)
	      IFUNC_IMPL_ADD (array, i, __mempcpy_chk,
			      CPU_FEATURE_USABLE (SSE2),
			      __mempcpy_chk_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, __mempcpy_chk, 1,
			      __mempcpy_chk_ia32))

  /* Support sysdeps/i386/i686/multiarch/mempcpy.S.  */
  IFUNC_IMPL (i, name, mempcpy,
	      IFUNC_IMPL_ADD (array, i, mempcpy, CPU_FEATURE_USABLE (SSSE3),
			      __mempcpy_ssse3_rep)
	      IFUNC_IMPL_ADD (array, i, mempcpy, CPU_FEATURE_USABLE (SSSE3),
			      __mempcpy_ssse3)
	      IFUNC_IMPL_ADD (array, i, mempcpy, CPU_FEATURE_USABLE (SSE2),
			      __mempcpy_sse2_unaligned)
	      IFUNC_IMPL_ADD (array, i, mempcpy, 1, __mempcpy_ia32))

  /* Support sysdeps/i386/i686/multiarch/strlen.S.  */
  IFUNC_IMPL (i, name, strlen,
	      IFUNC_IMPL_ADD (array, i, strlen, CPU_FEATURE_USABLE (SSE2),
			      __strlen_sse2_bsf)
	      IFUNC_IMPL_ADD (array, i, strlen, CPU_FEATURE_USABLE (SSE2),
			      __strlen_sse2)
	      IFUNC_IMPL_ADD (array, i, strlen, 1, __strlen_ia32))

  /* Support sysdeps/i386/i686/multiarch/strncmp.S.  */
  IFUNC_IMPL (i, name, strncmp,
	      IFUNC_IMPL_ADD (array, i, strncmp, CPU_FEATURE_USABLE (SSE4_2),
			      __strncmp_sse4_2)
	      IFUNC_IMPL_ADD (array, i, strncmp, CPU_FEATURE_USABLE (SSSE3),
			      __strncmp_ssse3)
	      IFUNC_IMPL_ADD (array, i, strncmp, 1, __strncmp_ia32))
#endif

  return i;
}
