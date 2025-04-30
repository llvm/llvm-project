/* Enumerate available IFUNC implementations of a function. s390/s390x version.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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
#include <ifunc-resolve.h>
#include <ifunc-memset.h>
#include <ifunc-memcmp.h>
#include <ifunc-memcpy.h>
#include <ifunc-strstr.h>
#include <ifunc-memmem.h>
#include <ifunc-strlen.h>
#include <ifunc-strnlen.h>
#include <ifunc-strcpy.h>
#include <ifunc-stpcpy.h>
#include <ifunc-strncpy.h>
#include <ifunc-stpncpy.h>
#include <ifunc-strcat.h>
#include <ifunc-strncat.h>
#include <ifunc-strcmp.h>
#include <ifunc-strncmp.h>
#include <ifunc-strchr.h>
#include <ifunc-strchrnul.h>
#include <ifunc-strrchr.h>
#include <ifunc-strspn.h>
#include <ifunc-strpbrk.h>
#include <ifunc-strcspn.h>
#include <ifunc-memchr.h>
#include <ifunc-rawmemchr.h>
#include <ifunc-memccpy.h>
#include <ifunc-memrchr.h>
#include <ifunc-wcslen.h>
#include <ifunc-wcsnlen.h>
#include <ifunc-wcscpy.h>
#include <ifunc-wcpcpy.h>
#include <ifunc-wcsncpy.h>
#include <ifunc-wcpncpy.h>
#include <ifunc-wcscat.h>
#include <ifunc-wcsncat.h>
#include <ifunc-wcscmp.h>
#include <ifunc-wcsncmp.h>
#include <ifunc-wcschr.h>
#include <ifunc-wcschrnul.h>
#include <ifunc-wcsrchr.h>
#include <ifunc-wcsspn.h>
#include <ifunc-wcspbrk.h>
#include <ifunc-wcscspn.h>
#include <ifunc-wmemchr.h>
#include <ifunc-wmemset.h>
#include <ifunc-wmemcmp.h>

/* Maximum number of IFUNC implementations.  */
#define MAX_IFUNC	3

/* Fill ARRAY of MAX elements with IFUNC implementations for function
   NAME supported on target machine and return the number of valid
   entries.  */
size_t
__libc_ifunc_impl_list (const char *name, struct libc_ifunc_impl *array,
			size_t max)
{
  assert (max >= MAX_IFUNC);

  size_t i = 0;

  /* Get hardware information.  */
  unsigned long int dl_hwcap = GLRO (dl_hwcap);
  unsigned long long stfle_bits = 0ULL;
  if ((dl_hwcap & HWCAP_S390_STFLE)
	&& (dl_hwcap & HWCAP_S390_ZARCH)
	&& (dl_hwcap & HWCAP_S390_HIGH_GPRS))
    {
      S390_STORE_STFLE (stfle_bits);
    }

#if HAVE_MEMSET_IFUNC
  IFUNC_IMPL (i, name, memset,
# if HAVE_MEMSET_Z196
	      IFUNC_IMPL_ADD (array, i, memset,
			      S390_IS_Z196 (stfle_bits), MEMSET_Z196)
# endif
# if HAVE_MEMSET_Z10
	      IFUNC_IMPL_ADD (array, i, memset,
			      S390_IS_Z10 (stfle_bits), MEMSET_Z10)
# endif
# if HAVE_MEMSET_Z900_G5
	      IFUNC_IMPL_ADD (array, i, memset, 1, MEMSET_Z900_G5)
# endif
	      )

  /* Note: bzero is implemented in memset.  */
  IFUNC_IMPL (i, name, bzero,
# if HAVE_MEMSET_Z196
	      IFUNC_IMPL_ADD (array, i, bzero,
			      S390_IS_Z196 (stfle_bits), BZERO_Z196)
# endif
# if HAVE_MEMSET_Z10
	      IFUNC_IMPL_ADD (array, i, bzero,
			      S390_IS_Z10 (stfle_bits), BZERO_Z10)
# endif
# if HAVE_MEMSET_Z900_G5
	      IFUNC_IMPL_ADD (array, i, bzero, 1, BZERO_Z900_G5)
# endif
	      )
#endif /* HAVE_MEMSET_IFUNC */

#if HAVE_MEMCMP_IFUNC
  IFUNC_IMPL (i, name, memcmp,
# if HAVE_MEMCMP_Z196
	      IFUNC_IMPL_ADD (array, i, memcmp,
			      S390_IS_Z196 (stfle_bits), MEMCMP_Z196)
# endif
# if HAVE_MEMCMP_Z10
	      IFUNC_IMPL_ADD (array, i, memcmp,
			      S390_IS_Z10 (stfle_bits), MEMCMP_Z10)
# endif
# if HAVE_MEMCMP_Z900_G5
	      IFUNC_IMPL_ADD (array, i, memcmp, 1, MEMCMP_Z900_G5)
# endif
	      )
#endif /* HAVE_MEMCMP_IFUNC */

#if HAVE_MEMCPY_IFUNC
  IFUNC_IMPL (i, name, memcpy,
# if HAVE_MEMCPY_Z196
	      IFUNC_IMPL_ADD (array, i, memcpy,
			      S390_IS_Z196 (stfle_bits), MEMCPY_Z196)
# endif
# if HAVE_MEMCPY_Z10
	      IFUNC_IMPL_ADD (array, i, memcpy,
			      S390_IS_Z10 (stfle_bits), MEMCPY_Z10)
# endif
# if HAVE_MEMCPY_Z900_G5
	      IFUNC_IMPL_ADD (array, i, memcpy, 1, MEMCPY_Z900_G5)
# endif
	      )

  IFUNC_IMPL (i, name, mempcpy,
# if HAVE_MEMCPY_Z196
	      IFUNC_IMPL_ADD (array, i, mempcpy,
			      S390_IS_Z196 (stfle_bits), MEMPCPY_Z196)
# endif
# if HAVE_MEMCPY_Z10
	      IFUNC_IMPL_ADD (array, i, mempcpy,
			      S390_IS_Z10 (stfle_bits), MEMPCPY_Z10)
# endif
# if HAVE_MEMCPY_Z900_G5
	      IFUNC_IMPL_ADD (array, i, mempcpy, 1, MEMPCPY_Z900_G5)
# endif
	      )
#endif /* HAVE_MEMCPY_IFUNC  */

#if HAVE_MEMMOVE_IFUNC
    IFUNC_IMPL (i, name, memmove,
# if HAVE_MEMMOVE_ARCH13
		IFUNC_IMPL_ADD (array, i, memmove,
				((dl_hwcap & HWCAP_S390_VXRS_EXT2)
				 && S390_IS_ARCH13_MIE3 (stfle_bits)),
				MEMMOVE_ARCH13)
# endif
# if HAVE_MEMMOVE_Z13
		IFUNC_IMPL_ADD (array, i, memmove,
				dl_hwcap & HWCAP_S390_VX, MEMMOVE_Z13)
# endif
# if HAVE_MEMMOVE_C
		IFUNC_IMPL_ADD (array, i, memmove, 1, MEMMOVE_C)
# endif
		)
#endif /* HAVE_MEMMOVE_IFUNC  */

#if HAVE_STRSTR_IFUNC
    IFUNC_IMPL (i, name, strstr,
# if HAVE_STRSTR_ARCH13
		IFUNC_IMPL_ADD (array, i, strstr,
				dl_hwcap & HWCAP_S390_VXRS_EXT2, STRSTR_ARCH13)
# endif
# if HAVE_STRSTR_Z13
		IFUNC_IMPL_ADD (array, i, strstr,
				dl_hwcap & HWCAP_S390_VX, STRSTR_Z13)
# endif
# if HAVE_STRSTR_C
		IFUNC_IMPL_ADD (array, i, strstr, 1, STRSTR_C)
# endif
		)
#endif /* HAVE_STRSTR_IFUNC  */

#if HAVE_MEMMEM_IFUNC
    IFUNC_IMPL (i, name, memmem,
# if HAVE_MEMMEM_ARCH13
	      IFUNC_IMPL_ADD (array, i, memmem,
			      dl_hwcap & HWCAP_S390_VXRS_EXT2, MEMMEM_ARCH13)
# endif
# if HAVE_MEMMEM_Z13
		IFUNC_IMPL_ADD (array, i, memmem,
				dl_hwcap & HWCAP_S390_VX, MEMMEM_Z13)
# endif
# if HAVE_MEMMEM_C
		IFUNC_IMPL_ADD (array, i, memmem, 1, MEMMEM_C)
# endif
		)
#endif /* HAVE_MEMMEM_IFUNC  */

#if HAVE_STRLEN_IFUNC
    IFUNC_IMPL (i, name, strlen,
# if HAVE_STRLEN_Z13
		IFUNC_IMPL_ADD (array, i, strlen,
				dl_hwcap & HWCAP_S390_VX, STRLEN_Z13)
# endif
# if HAVE_STRLEN_C
		IFUNC_IMPL_ADD (array, i, strlen, 1, STRLEN_C)
# endif
		)
#endif /* HAVE_STRLEN_IFUNC  */

#if HAVE_STRNLEN_IFUNC
    IFUNC_IMPL (i, name, strnlen,
# if HAVE_STRNLEN_Z13
		IFUNC_IMPL_ADD (array, i, strnlen,
				dl_hwcap & HWCAP_S390_VX, STRNLEN_Z13)
# endif
# if HAVE_STRNLEN_C
		IFUNC_IMPL_ADD (array, i, strnlen, 1, STRNLEN_C)
# endif
		)
#endif /* HAVE_STRNLEN_IFUNC  */

#if HAVE_STRCPY_IFUNC
    IFUNC_IMPL (i, name, strcpy,
# if HAVE_STRCPY_Z13
		IFUNC_IMPL_ADD (array, i, strcpy,
				dl_hwcap & HWCAP_S390_VX, STRCPY_Z13)
# endif
# if HAVE_STRCPY_Z900_G5
		IFUNC_IMPL_ADD (array, i, strcpy, 1, STRCPY_Z900_G5)
# endif
		)
#endif /* HAVE_STRCPY_IFUNC  */

#if HAVE_STPCPY_IFUNC
    IFUNC_IMPL (i, name, stpcpy,
# if HAVE_STPCPY_Z13
		IFUNC_IMPL_ADD (array, i, stpcpy,
				dl_hwcap & HWCAP_S390_VX, STPCPY_Z13)
# endif
# if HAVE_STPCPY_C
		IFUNC_IMPL_ADD (array, i, stpcpy, 1, STPCPY_C)
# endif
		)
#endif /* HAVE_STPCPY_IFUNC  */

#if HAVE_STRNCPY_IFUNC
    IFUNC_IMPL (i, name, strncpy,
# if HAVE_STRNCPY_Z13
		IFUNC_IMPL_ADD (array, i, strncpy,
				dl_hwcap & HWCAP_S390_VX, STRNCPY_Z13)
# endif
# if HAVE_STRNCPY_Z900_G5
		IFUNC_IMPL_ADD (array, i, strncpy, 1, STRNCPY_Z900_G5)
# endif
		)
#endif /* HAVE_STRNCPY_IFUNC  */

#if HAVE_STPNCPY_IFUNC
    IFUNC_IMPL (i, name, stpncpy,
# if HAVE_STPNCPY_Z13
		IFUNC_IMPL_ADD (array, i, stpncpy,
				dl_hwcap & HWCAP_S390_VX, STPNCPY_Z13)
# endif
# if HAVE_STPNCPY_C
		IFUNC_IMPL_ADD (array, i, stpncpy, 1, STPNCPY_C)
# endif
		)
#endif /* HAVE_STPNCPY_IFUNC  */

#if HAVE_STRCAT_IFUNC
    IFUNC_IMPL (i, name, strcat,
# if HAVE_STRCAT_Z13
		IFUNC_IMPL_ADD (array, i, strcat,
				dl_hwcap & HWCAP_S390_VX, STRCAT_Z13)
# endif
# if HAVE_STRCAT_C
		IFUNC_IMPL_ADD (array, i, strcat, 1, STRCAT_C)
# endif
		)
#endif /* HAVE_STRCAT_IFUNC  */

#if HAVE_STRNCAT_IFUNC
    IFUNC_IMPL (i, name, strncat,
# if HAVE_STRNCAT_Z13
		IFUNC_IMPL_ADD (array, i, strncat,
				dl_hwcap & HWCAP_S390_VX, STRNCAT_Z13)
# endif
# if HAVE_STRNCAT_C
		IFUNC_IMPL_ADD (array, i, strncat, 1, STRNCAT_C)
# endif
		)
#endif /* HAVE_STRNCAT_IFUNC  */

#if HAVE_STRCMP_IFUNC
    IFUNC_IMPL (i, name, strcmp,
# if HAVE_STRCMP_Z13
		IFUNC_IMPL_ADD (array, i, strcmp,
				dl_hwcap & HWCAP_S390_VX, STRCMP_Z13)
# endif
# if HAVE_STRCMP_Z900_G5
		IFUNC_IMPL_ADD (array, i, strcmp, 1, STRCMP_Z900_G5)
# endif
		)
#endif /* HAVE_STRCMP_IFUNC  */

#if HAVE_STRNCMP_IFUNC
    IFUNC_IMPL (i, name, strncmp,
# if HAVE_STRNCMP_Z13
		IFUNC_IMPL_ADD (array, i, strncmp,
				dl_hwcap & HWCAP_S390_VX, STRNCMP_Z13)
# endif
# if HAVE_STRNCMP_C
		IFUNC_IMPL_ADD (array, i, strncmp, 1, STRNCMP_C)
# endif
		)
#endif /* HAVE_STRNCMP_IFUNC  */

#if HAVE_STRCHR_IFUNC
    IFUNC_IMPL (i, name, strchr,
# if HAVE_STRCHR_Z13
		IFUNC_IMPL_ADD (array, i, strchr,
				dl_hwcap & HWCAP_S390_VX, STRCHR_Z13)
# endif
# if HAVE_STRCHR_C
		IFUNC_IMPL_ADD (array, i, strchr, 1, STRCHR_C)
# endif
		)
#endif /* HAVE_STRCHR_IFUNC  */

#if HAVE_STRCHRNUL_IFUNC
    IFUNC_IMPL (i, name, strchrnul,
# if HAVE_STRCHRNUL_Z13
		IFUNC_IMPL_ADD (array, i, strchrnul,
				dl_hwcap & HWCAP_S390_VX, STRCHRNUL_Z13)
# endif
# if HAVE_STRCHRNUL_C
		IFUNC_IMPL_ADD (array, i, strchrnul, 1, STRCHRNUL_C)
# endif
		)
#endif /* HAVE_STRCHRNUL_IFUNC  */

#if HAVE_STRRCHR_IFUNC
    IFUNC_IMPL (i, name, strrchr,
# if HAVE_STRRCHR_Z13
		IFUNC_IMPL_ADD (array, i, strrchr,
				dl_hwcap & HWCAP_S390_VX, STRRCHR_Z13)
# endif
# if HAVE_STRRCHR_C
		IFUNC_IMPL_ADD (array, i, strrchr, 1, STRRCHR_C)
# endif
		)
#endif /* HAVE_STRRCHR_IFUNC  */

#if HAVE_STRSPN_IFUNC
    IFUNC_IMPL (i, name, strspn,
# if HAVE_STRSPN_Z13
		IFUNC_IMPL_ADD (array, i, strspn,
				dl_hwcap & HWCAP_S390_VX, STRSPN_Z13)
# endif
# if HAVE_STRSPN_C
		IFUNC_IMPL_ADD (array, i, strspn, 1, STRSPN_C)
# endif
		)
#endif /* HAVE_STRSPN_IFUNC  */

#if HAVE_STRPBRK_IFUNC
    IFUNC_IMPL (i, name, strpbrk,
# if HAVE_STRPBRK_Z13
		IFUNC_IMPL_ADD (array, i, strpbrk,
				dl_hwcap & HWCAP_S390_VX, STRPBRK_Z13)
# endif
# if HAVE_STRPBRK_C
		IFUNC_IMPL_ADD (array, i, strpbrk, 1, STRPBRK_C)
# endif
		)
#endif /* HAVE_STRPBRK_IFUNC  */

#if HAVE_STRCSPN_IFUNC
    IFUNC_IMPL (i, name, strcspn,
# if HAVE_STRCSPN_Z13
		IFUNC_IMPL_ADD (array, i, strcspn,
				dl_hwcap & HWCAP_S390_VX, STRCSPN_Z13)
# endif
# if HAVE_STRCSPN_C
		IFUNC_IMPL_ADD (array, i, strcspn, 1, STRCSPN_C)
# endif
		)
#endif /* HAVE_STRCSPN_IFUNC  */

#if HAVE_MEMCHR_IFUNC
    IFUNC_IMPL (i, name, memchr,
# if HAVE_MEMCHR_Z13
		IFUNC_IMPL_ADD (array, i, memchr,
				dl_hwcap & HWCAP_S390_VX, MEMCHR_Z13)
# endif
# if HAVE_MEMCHR_Z900_G5
		IFUNC_IMPL_ADD (array, i, memchr, 1, MEMCHR_Z900_G5)
# endif
		)
#endif /* HAVE_MEMCHR_IFUNC  */

#if HAVE_RAWMEMCHR_IFUNC
    IFUNC_IMPL (i, name, rawmemchr,
# if HAVE_RAWMEMCHR_Z13
		IFUNC_IMPL_ADD (array, i, rawmemchr,
				dl_hwcap & HWCAP_S390_VX, RAWMEMCHR_Z13)
# endif
# if HAVE_RAWMEMCHR_C
		IFUNC_IMPL_ADD (array, i, rawmemchr, 1, RAWMEMCHR_C)
# endif
		)
#endif /* HAVE_RAWMEMCHR_IFUNC  */

#if HAVE_MEMCCPY_IFUNC
    IFUNC_IMPL (i, name, memccpy,
# if HAVE_MEMCCPY_Z13
		IFUNC_IMPL_ADD (array, i, memccpy,
				dl_hwcap & HWCAP_S390_VX, MEMCCPY_Z13)
# endif
# if HAVE_MEMCCPY_C
		IFUNC_IMPL_ADD (array, i, memccpy, 1, MEMCCPY_C)
# endif
		)
#endif /* HAVE_MEMCCPY_IFUNC  */

#if HAVE_MEMRCHR_IFUNC
    IFUNC_IMPL (i, name, memrchr,
# if HAVE_MEMRCHR_Z13
		IFUNC_IMPL_ADD (array, i, memrchr,
				dl_hwcap & HWCAP_S390_VX, MEMRCHR_Z13)
# endif
# if HAVE_MEMRCHR_C
		IFUNC_IMPL_ADD (array, i, memrchr, 1, MEMRCHR_C)
# endif
		)
#endif /* HAVE_MEMRCHR_IFUNC  */

#if HAVE_WCSLEN_IFUNC
    IFUNC_IMPL (i, name, wcslen,
# if HAVE_WCSLEN_Z13
		IFUNC_IMPL_ADD (array, i, wcslen,
				dl_hwcap & HWCAP_S390_VX, WCSLEN_Z13)
# endif
# if HAVE_WCSLEN_C
		IFUNC_IMPL_ADD (array, i, wcslen, 1, WCSLEN_C)
# endif
		)
#endif /* HAVE_WCSLEN_IFUNC  */

#if HAVE_WCSNLEN_IFUNC
    IFUNC_IMPL (i, name, wcsnlen,
# if HAVE_WCSNLEN_Z13
		IFUNC_IMPL_ADD (array, i, wcsnlen,
				dl_hwcap & HWCAP_S390_VX, WCSNLEN_Z13)
# endif
# if HAVE_WCSNLEN_C
		IFUNC_IMPL_ADD (array, i, wcsnlen, 1, WCSNLEN_C)
# endif
		)
#endif /* HAVE_WCSNLEN_IFUNC  */

#if HAVE_WCSCPY_IFUNC
    IFUNC_IMPL (i, name, wcscpy,
# if HAVE_WCSCPY_Z13
		IFUNC_IMPL_ADD (array, i, wcscpy,
				dl_hwcap & HWCAP_S390_VX, WCSCPY_Z13)
# endif
# if HAVE_WCSCPY_C
		IFUNC_IMPL_ADD (array, i, wcscpy, 1, WCSCPY_C)
# endif
		)
#endif /* HAVE_WCSCPY_IFUNC  */

#if HAVE_WCPCPY_IFUNC
    IFUNC_IMPL (i, name, wcpcpy,
# if HAVE_WCPCPY_Z13
		IFUNC_IMPL_ADD (array, i, wcpcpy,
				dl_hwcap & HWCAP_S390_VX, WCPCPY_Z13)
# endif
# if HAVE_WCPCPY_C
		IFUNC_IMPL_ADD (array, i, wcpcpy, 1, WCPCPY_C)
# endif
		)
#endif /* HAVE_WCPCPY_IFUNC  */

#if HAVE_WCSNCPY_IFUNC
    IFUNC_IMPL (i, name, wcsncpy,
# if HAVE_WCSNCPY_Z13
		IFUNC_IMPL_ADD (array, i, wcsncpy,
				dl_hwcap & HWCAP_S390_VX, WCSNCPY_Z13)
# endif
# if HAVE_WCSNCPY_C
		IFUNC_IMPL_ADD (array, i, wcsncpy, 1, WCSNCPY_C)
# endif
		)
#endif /* HAVE_WCSNCPY_IFUNC  */

#if HAVE_WCPNCPY_IFUNC
    IFUNC_IMPL (i, name, wcpncpy,
# if HAVE_WCPNCPY_Z13
		IFUNC_IMPL_ADD (array, i, wcpncpy,
				dl_hwcap & HWCAP_S390_VX, WCPNCPY_Z13)
# endif
# if HAVE_WCPNCPY_C
		IFUNC_IMPL_ADD (array, i, wcpncpy, 1, WCPNCPY_C)
# endif
		)
#endif /* HAVE_WCPNCPY_IFUNC  */

#if HAVE_WCSCAT_IFUNC
    IFUNC_IMPL (i, name, wcscat,
# if HAVE_WCSCAT_Z13
		IFUNC_IMPL_ADD (array, i, wcscat,
				dl_hwcap & HWCAP_S390_VX, WCSCAT_Z13)
# endif
# if HAVE_WCSCAT_C
		IFUNC_IMPL_ADD (array, i, wcscat, 1, WCSCAT_C)
# endif
		)
#endif /* HAVE_WCSCAT_IFUNC  */

#if HAVE_WCSNCAT_IFUNC
    IFUNC_IMPL (i, name, wcsncat,
# if HAVE_WCSNCAT_Z13
		IFUNC_IMPL_ADD (array, i, wcsncat,
				dl_hwcap & HWCAP_S390_VX, WCSNCAT_Z13)
# endif
# if HAVE_WCSNCAT_C
		IFUNC_IMPL_ADD (array, i, wcsncat, 1, WCSNCAT_C)
# endif
		)
#endif /* HAVE_WCSNCAT_IFUNC  */

#if HAVE_WCSCMP_IFUNC
    IFUNC_IMPL (i, name, wcscmp,
# if HAVE_WCSCMP_Z13
		IFUNC_IMPL_ADD (array, i, wcscmp,
				dl_hwcap & HWCAP_S390_VX, WCSCMP_Z13)
# endif
# if HAVE_WCSCMP_C
		IFUNC_IMPL_ADD (array, i, wcscmp, 1, WCSCMP_C)
# endif
		)
#endif /* HAVE_WCSCMP_IFUNC  */

#if HAVE_WCSNCMP_IFUNC
    IFUNC_IMPL (i, name, wcsncmp,
# if HAVE_WCSNCMP_Z13
		IFUNC_IMPL_ADD (array, i, wcsncmp,
				dl_hwcap & HWCAP_S390_VX, WCSNCMP_Z13)
# endif
# if HAVE_WCSNCMP_C
		IFUNC_IMPL_ADD (array, i, wcsncmp, 1, WCSNCMP_C)
# endif
		)
#endif /* HAVE_WCSNCMP_IFUNC  */

#if HAVE_WCSCHR_IFUNC
    IFUNC_IMPL (i, name, wcschr,
# if HAVE_WCSCHR_Z13
		IFUNC_IMPL_ADD (array, i, wcschr,
				dl_hwcap & HWCAP_S390_VX, WCSCHR_Z13)
# endif
# if HAVE_WCSCHR_C
		IFUNC_IMPL_ADD (array, i, wcschr, 1, WCSCHR_C)
# endif
		)
#endif /* HAVE_WCSCHR_IFUNC  */

#if HAVE_WCSCHRNUL_IFUNC
    IFUNC_IMPL (i, name, wcschrnul,
# if HAVE_WCSCHRNUL_Z13
		IFUNC_IMPL_ADD (array, i, wcschrnul,
				dl_hwcap & HWCAP_S390_VX, WCSCHRNUL_Z13)
# endif
# if HAVE_WCSCHRNUL_C
		IFUNC_IMPL_ADD (array, i, wcschrnul, 1, WCSCHRNUL_C)
# endif
		)
#endif /* HAVE_WCSCHRNUL_IFUNC  */

#if HAVE_WCSRCHR_IFUNC
    IFUNC_IMPL (i, name, wcsrchr,
# if HAVE_WCSRCHR_Z13
		IFUNC_IMPL_ADD (array, i, wcsrchr,
				dl_hwcap & HWCAP_S390_VX, WCSRCHR_Z13)
# endif
# if HAVE_WCSRCHR_C
		IFUNC_IMPL_ADD (array, i, wcsrchr, 1, WCSRCHR_C)
# endif
		)
#endif /* HAVE_WCSRCHR_IFUNC  */

#if HAVE_WCSSPN_IFUNC
    IFUNC_IMPL (i, name, wcsspn,
# if HAVE_WCSSPN_Z13
		IFUNC_IMPL_ADD (array, i, wcsspn,
				dl_hwcap & HWCAP_S390_VX, WCSSPN_Z13)
# endif
# if HAVE_WCSSPN_C
		IFUNC_IMPL_ADD (array, i, wcsspn, 1, WCSSPN_C)
# endif
		)
#endif /* HAVE_WCSSPN_IFUNC  */

#if HAVE_WCSPBRK_IFUNC
    IFUNC_IMPL (i, name, wcspbrk,
# if HAVE_WCSPBRK_Z13
		IFUNC_IMPL_ADD (array, i, wcspbrk,
				dl_hwcap & HWCAP_S390_VX, WCSPBRK_Z13)
# endif
# if HAVE_WCSPBRK_C
		IFUNC_IMPL_ADD (array, i, wcspbrk, 1, WCSPBRK_C)
# endif
		)
#endif /* HAVE_WCSPBRK_IFUNC  */

#if HAVE_WCSCSPN_IFUNC
    IFUNC_IMPL (i, name, wcscspn,
# if HAVE_WCSCSPN_Z13
		IFUNC_IMPL_ADD (array, i, wcscspn,
				dl_hwcap & HWCAP_S390_VX, WCSCSPN_Z13)
# endif
# if HAVE_WCSCSPN_C
		IFUNC_IMPL_ADD (array, i, wcscspn, 1, WCSCSPN_C)
# endif
		)
#endif /* HAVE_WCSCSPN_IFUNC  */

#if HAVE_WMEMCHR_IFUNC
    IFUNC_IMPL (i, name, wmemchr,
# if HAVE_WMEMCHR_Z13
		IFUNC_IMPL_ADD (array, i, wmemchr,
				dl_hwcap & HWCAP_S390_VX, WMEMCHR_Z13)
# endif
# if HAVE_WMEMCHR_C
		IFUNC_IMPL_ADD (array, i, wmemchr, 1, WMEMCHR_C)
# endif
		)
#endif /* HAVE_WMEMCHR_IFUNC  */

#if HAVE_WMEMSET_IFUNC
    IFUNC_IMPL (i, name, wmemset,
# if HAVE_WMEMSET_Z13
		IFUNC_IMPL_ADD (array, i, wmemset,
				dl_hwcap & HWCAP_S390_VX, WMEMSET_Z13)
# endif
# if HAVE_WMEMSET_C
		IFUNC_IMPL_ADD (array, i, wmemset, 1, WMEMSET_C)
# endif
		)
#endif /* HAVE_WMEMSET_IFUNC  */

#if HAVE_WMEMCMP_IFUNC
    IFUNC_IMPL (i, name, wmemcmp,
# if HAVE_WMEMCMP_Z13
		IFUNC_IMPL_ADD (array, i, wmemcmp,
				dl_hwcap & HWCAP_S390_VX, WMEMCMP_Z13)
# endif
# if HAVE_WMEMCMP_C
		IFUNC_IMPL_ADD (array, i, wmemcmp, 1, WMEMCMP_C)
# endif
		)
#endif /* HAVE_WMEMCMP_IFUNC  */

  return i;
}
